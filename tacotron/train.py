import argparse
import os
import subprocess
import time
import traceback
from datetime import datetime

import infolog
import numpy as np
import tensorflow as tf
from datasets import audio
from hparams import hparams_debug_string
from tacotron.feeder import Feeder
from tacotron.models import create_model
from tacotron.utils import ValueWindow, plot
from tacotron.utils.text import sequence_to_text
from tacotron.utils.symbols import symbols
from tqdm import tqdm

log = infolog.log

import mem_draw_util
_NUM_STEPS_TO_PROFILE = (30, 31)


def time_string():
  return datetime.now().strftime('%Y-%m-%d %H:%M')

def add_embedding_stats(summary_writer, embedding_names, paths_to_meta, checkpoint_path):
  #Create tensorboard projector
  config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
  config.model_checkpoint_path = checkpoint_path

  for embedding_name, path_to_meta in zip(embedding_names, paths_to_meta):
    #Initialize config
    embedding = config.embeddings.add()
    #Specifiy the embedding variable and the metadata
    embedding.tensor_name = embedding_name
    embedding.metadata_path = path_to_meta
  
  #Project the embeddings to space dimensions for visualization
  tf.contrib.tensorboard.plugins.projector.visualize_embeddings(summary_writer, config)

def add_train_stats(model, hparams):
  with tf.variable_scope('stats') as scope:
    for i in range(hparams.tacotron_num_gpus):
      tf.summary.histogram('mel_outputs %d' % i, model.tower_mel_outputs[i])
      tf.summary.histogram('mel_targets %d' % i, model.tower_mel_targets[i])
    tf.summary.scalar('before_loss', model.before_loss)
    tf.summary.scalar('after_loss', model.after_loss)

    if hparams.predict_linear:
      tf.summary.scalar('linear_loss', model.linear_loss)
      for i in range(hparams.tacotron_num_gpus):
        tf.summary.histogram('linear_outputs %d' % i, model.tower_linear_outputs[i])
        tf.summary.histogram('linear_targets %d' % i, model.tower_linear_targets[i])
    
    tf.summary.scalar('regularization_loss', model.regularization_loss)
    tf.summary.scalar('stop_token_loss', model.stop_token_loss)
    tf.summary.scalar('loss', model.loss)
    tf.summary.scalar('learning_rate', model.learning_rate) #Control learning rate decay speed
    if hparams.tacotron_teacher_forcing_mode == 'scheduled':
      tf.summary.scalar('teacher_forcing_ratio', model.ratio) #Control teacher forcing ratio decay when mode = 'scheduled'
    gradient_norms = [tf.norm(grad) for grad in model.gradients]
    tf.summary.histogram('gradient_norm', gradient_norms)
    tf.summary.scalar('max_gradient_norm', tf.reduce_max(gradient_norms)) #visualize gradients (in case of explosion)
    return tf.summary.merge_all()

def add_eval_stats(summary_writer, step, linear_loss, before_loss, after_loss, stop_token_loss, loss):
  values = [
  tf.Summary.Value(tag='Tacotron_eval_model/eval_stats/eval_before_loss', simple_value=before_loss),
  tf.Summary.Value(tag='Tacotron_eval_model/eval_stats/eval_after_loss', simple_value=after_loss),
  tf.Summary.Value(tag='Tacotron_eval_model/eval_stats/stop_token_loss', simple_value=stop_token_loss),
  tf.Summary.Value(tag='Tacotron_eval_model/eval_stats/eval_loss', simple_value=loss),
  ]
  if linear_loss is not None:
    values.append(tf.Summary.Value(tag='Tacotron_eval_model/eval_stats/eval_linear_loss', simple_value=linear_loss))
  test_summary = tf.Summary(value=values)
  summary_writer.add_summary(test_summary, step)

def model_train_mode(args, feeder, hparams, global_step):
  with tf.variable_scope('Tacotron_model', reuse=tf.AUTO_REUSE) as scope:
    model_name = None
    if args.model == 'Tacotron-2':
      model_name = 'Tacotron'
    model = create_model(model_name or args.model, hparams)
    if hparams.predict_linear:
      model.initialize(feeder.inputs, feeder.input_lengths, feeder.mel_targets, feeder.token_targets, linear_targets=feeder.linear_targets,
        targets_lengths=feeder.targets_lengths, global_step=global_step,
        is_training=True, split_infos=feeder.split_infos)
    else:
      model.initialize(feeder.inputs, feeder.input_lengths, feeder.mel_targets, feeder.token_targets,
        targets_lengths=feeder.targets_lengths, global_step=global_step,
        is_training=True, split_infos=feeder.split_infos)
    model.add_loss()
    model.add_optimizer(global_step)
    stats = add_train_stats(model, hparams)
    return model, stats

def model_test_mode(args, feeder, hparams, global_step):
  with tf.variable_scope('Tacotron_model', reuse=tf.AUTO_REUSE) as scope:
    model_name = None
    if args.model == 'Tacotron-2':
      model_name = 'Tacotron'
    model = create_model(model_name or args.model, hparams)
    if hparams.predict_linear:
      model.initialize(feeder.eval_inputs, feeder.eval_input_lengths, feeder.eval_mel_targets, feeder.eval_token_targets,
        linear_targets=feeder.eval_linear_targets, targets_lengths=feeder.eval_targets_lengths, global_step=global_step,
        is_training=False, is_evaluating=True, split_infos=feeder.eval_split_infos)
    else:
      model.initialize(feeder.eval_inputs, feeder.eval_input_lengths, feeder.eval_mel_targets, feeder.eval_token_targets,
        targets_lengths=feeder.eval_targets_lengths, global_step=global_step, is_training=False, is_evaluating=True, 
        split_infos=feeder.eval_split_infos)
    model.add_loss()
    return model

def train(log_dir, args, hparams):
  save_dir = os.path.join(log_dir, 'taco_pretrained')
  plot_dir = os.path.join(log_dir, 'plots')
  wav_dir = os.path.join(log_dir, 'wavs')
  mel_dir = os.path.join(log_dir, 'mel-spectrograms')
  eval_dir = os.path.join(log_dir, 'eval-dir')
  eval_plot_dir = os.path.join(eval_dir, 'plots')
  eval_wav_dir = os.path.join(eval_dir, 'wavs')
  tensorboard_dir = os.path.join(log_dir, 'tacotron_events')
  meta_folder = os.path.join(log_dir, 'metas')
  os.makedirs(save_dir, exist_ok=True)
  os.makedirs(plot_dir, exist_ok=True)
  os.makedirs(wav_dir, exist_ok=True)
  os.makedirs(mel_dir, exist_ok=True)
  os.makedirs(eval_dir, exist_ok=True)
  os.makedirs(eval_plot_dir, exist_ok=True)
  os.makedirs(eval_wav_dir, exist_ok=True)
  os.makedirs(tensorboard_dir, exist_ok=True)
  os.makedirs(meta_folder, exist_ok=True)

  checkpoint_path = os.path.join(save_dir, 'tacotron_model.ckpt')
  input_path = os.path.join(args.base_dir, args.tacotron_input)
  #log(input_path)
  if hparams.predict_linear:
    linear_dir = os.path.join(log_dir, 'linear-spectrograms')
    os.makedirs(linear_dir, exist_ok=True)

  log('Checkpoint path: {}'.format(checkpoint_path))
  log('Loading training data from: {}'.format(input_path))
  log('Using model: {}'.format(args.model))
  log(hparams_debug_string())

  #Start by setting a seed for repeatability
  tf.set_random_seed(hparams.tacotron_random_seed)

  #Set up data feeder
  coord = tf.train.Coordinator()
  with tf.variable_scope('datafeeder') as scope:
    feeder = Feeder(coord, input_path, hparams)
  #log("input_path : {}".format(input_path))

  #Set up model:
  global_step = tf.Variable(0, name='global_step', trainable=False)
  model, stats = model_train_mode(args, feeder, hparams, global_step)
  eval_model = model_test_mode(args, feeder, hparams, global_step)

  #Embeddings metadata
  char_embedding_meta = os.path.join(meta_folder, 'CharacterEmbeddings.tsv')
  if not os.path.isfile(char_embedding_meta):
    with open(char_embedding_meta, 'w', encoding='utf-8') as f:
      for symbol in symbols:
        if symbol == ' ':
          symbol = '\\s' #For visual purposes, swap space with \s

        f.write('{}\n'.format(symbol))

  char_embedding_meta = char_embedding_meta.replace(log_dir, '..')

  #Potential Griffin-Lim GPU setup
  if hparams.GL_on_GPU:
    GLGPU_mel_inputs = tf.placeholder(tf.float32, (None, hparams.num_mels), name='GLGPU_mel_inputs')
    GLGPU_lin_inputs = tf.placeholder(tf.float32, (None, hparams.num_freq), name='GLGPU_lin_inputs')

    GLGPU_mel_outputs = audio.inv_mel_spectrogram_tensorflow(GLGPU_mel_inputs, hparams)
    GLGPU_lin_outputs = audio.inv_linear_spectrogram_tensorflow(GLGPU_lin_inputs, hparams)

  #Book keeping
  step = 0
  time_window = ValueWindow(100)
  loss_window = ValueWindow(100)
  saver = tf.train.Saver(max_to_keep=20)

  log('Tacotron training set to a maximum of {} steps'.format(args.tacotron_train_steps))

  #Memory allocation on the GPU as needed
  config = tf.compat.v1.ConfigProto()
  config.gpu_options.allow_growth = args.allow_growth
  config.graph_options.build_cost_model = args.build_cost_model
  config.graph_options.build_cost_model_after = args.build_cost_model_after
  config.gpu_options.allow_shared = args.allow_shared
  config.gpu_options.experimental.use_unified_memory = args.use_unified_memory
  if args.gpu_memory_frac_for_testing > 0:
    config.gpu_options.per_process_gpu_memory_fraction = args.gpu_memory_frac_for_testing
  config.allow_soft_placement = True

  #Train
  with tf.compat.v1.Session(target=args.target, config=config) as sess:
    try:
      summary_writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)
      sess.run(tf.global_variables_initializer())
      #coord = tf.compat.v1.train.Coordinator()
      #threads = tf.compat.v1.train.start_queue_runners(sess,coord)
      #saved model restoring
      if args.restore:
        # Restore saved model if the user requested it, default = True
        try:
          checkpoint_state = tf.train.get_checkpoint_state(save_dir)

          if (checkpoint_state and checkpoint_state.model_checkpoint_path):
            log('Loading checkpoint {}'.format(checkpoint_state.model_checkpoint_path), slack=True)
            saver.restore(sess, checkpoint_state.model_checkpoint_path)

          else:
            log('No model to load at {}'.format(save_dir), slack=True)
            saver.save(sess, checkpoint_path, global_step=global_step)

        except tf.errors.OutOfRangeError as e:
          log('Cannot restore checkpoint: {}'.format(e), slack=True)
      else:
        log('Starting new training!', slack=True)
        saver.save(sess, checkpoint_path, global_step=global_step)
      
     # initializing feeder
      #feeder.start_threads(sess)
      #log('there')
      #for i in range(2):
      #b_inputs = sess.run(model.Inputs2)
      #a_inputs ,a_inputs_shape,a_inputs_max,a_inputs_min, a_inputs_mean, a_tower_input_lengths ,a_tower_input_lengths_shape, a_mel_targets,a_mel_targets_shape,a_mel_targets_max,a_mel_targets_min,a_mel_targets_mean, a_stop_token_targets,a_stop_token_targets_shape, a_linear_targets,a_linear_targets_shape, a_linear_targets_max, a_linear_targets_min, a_linear_targets_mean, b_inputs ,b_inputs_shape,  b_mel_targets,b_mel_targets_shape, b_stop_token_targets,b_stop_token_targets_shape, b_linear_targets ,b_linear_targets_shape, p_inputs,p_inputs_shape, p_mel_targets,p_mel_targets_shape, p_stop_token_targets,p_stop_token_targets_shape, p_linear_targets,p_linear_targets_shape = sess.run([model.Inputs1, tf.shape(model.Inputs1),tf.reduce_max(model.Inputs1),tf.reduce_min(model.Inputs1),tf.reduce_mean(model.Inputs1),model.tower_input_lengths,tf.shape(model.tower_input_lengths),model.Mel_targets1,tf.shape(model.Mel_targets1), tf.reduce_max(model.Mel_targets1),tf.reduce_min(model.Mel_targets1),tf.reduce_mean(model.Mel_targets1),model.Stop_token_targets1 , tf.shape(model.Stop_token_targets1), model.Linear_targets1, tf.shape(model.Linear_targets1),tf.reduce_max(model.Linear_targets1), tf.reduce_min(model.Linear_targets1), tf.reduce_mean(model.Linear_targets1),  model.Inputs2 , tf.shape(model.Inputs2), model.Mel_targets2,  tf.shape(model.Mel_targets2), model.Stop_token_targets2, tf.shape(model.Stop_token_targets2), model.Linear_targets2,  tf.shape(model.Linear_targets2), model.tower_inputs, tf.shape(model.tower_inputs), model.tower_mel_targets, tf.shape(model.tower_mel_targets), model.tower_stop_token_targets, tf.shape(model.tower_stop_token_targets), model.tower_linear_targets, tf.shape(model.tower_linear_targets)])
      #   log('a_inputs :{}'.format(a_inputs))
      #log('here')
      #   log('a_inputs_shape :{}'.format(a_inputs_shape))
      #   log('a_inputs_max :{}'.format(a_inputs_max))
      #   log('a_inputs_min: {}'.format(a_inputs_min))
      #   log('a_inputs_mean: {}'.format(a_inputs_mean))
      #   log('a_tower_input_lengths : {}'.format(a_tower_input_lengths))
      #   log('a_tower_input_lengths : {}'.format(a_tower_input_lengths_shape))
      #   log('a_mel_targets :{}'.format(a_mel_targets))
      #   log('a_mel_targets_shape :{}'.format(a_mel_targets_shape))
      #   log('a_mel_targets_max :{}'.format(a_mel_targets_max))
      #   log('a_mel_targets_min: {}'.format(a_mel_targets_min))
      #   log('a_mel_targets_mean: {}'.format(a_mel_targets_mean))
      #   log('a_stop_token_targets :{}'.format(a_stop_token_targets))
      #   log('a_stop_token_targets_shape :{}'.format(a_stop_token_targets_shape))
      #   log('a_linear_targets :{}'.format(a_linear_targets))
      #   log('a_linear_targets_shape :{}'.format(a_linear_targets_shape))
      #   log('a_linear_targets_max :{}'.format(a_linear_targets_max))
      #   log('a_linear_targets_min: {}'.format(a_linear_targets_min))
      #   log('a_linear_targets_mean: {}'.format(a_linear_targets_mean))
     
      #   log('p_inputs :{}'.format(p_inputs))
      #   log('p_inputs_shape :{}'.format(p_inputs_shape))
      #   log('p_mel_targets :{}'.format(p_mel_targets))
      #   log('p_mel_targets_shape :{}'.format(p_mel_targets_shape))
      #   log('p_stop_token_targets :{}'.format(p_stop_token_targets))
      #   log('p_stop_token_targets_shape :{}'.format(p_stop_token_targets_shape))
      #   log('p_linear_targets :{}'.format(p_linear_targets))
      #   log('p_linear_targets_shape :{}'.format(p_linear_targets_shape))
      #feeder.start_threads(sess)
      
      #Input_length, Input_length_shape, target_length, target_length_shape, split_info, split_info_shape = sess.run([feeder.input_lengths,tf.shape(feeder.input_lengths),feeder.targets_lengths,tf.shape(feeder.targets_lengths),feeder.split_infos,tf.shape(feeder.split_infos)])
      #log('inputs_length :{}'.format(Input_length))
      #log('inputs_length_shape :{}'.format(Input_length_shape))
      #log('target_targets :{}'.format(target_length))
      #log('target_length_shape :{}'.format(target_length_shape))
      #log('split_info :{}'.format(split_info))
      #log('split_info_shape :{}'.format(split_info_shape))


      #Training loop
      speed = 0.0
      while not coord.should_stop() and step < args.tacotron_train_steps:
        if args.lognode_time:
          if _NUM_STEPS_TO_PROFILE[0] <= step < _NUM_STEPS_TO_PROFILE[1]:
            run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
            run_metadata = tf.compat.v1.RunMetadata()
          else:
            run_options = None
            run_metadata = None
        else:
          run_options = None
          run_metadata = None
        start_time = time.time()
        #queue_runner=tf.train.start_queue_runners(sess, coord)
        #thread = tf.train.start_queue_runners(sess,coord)
        #log("sess_run_begin")
        #for i in range(10):
        #print("global_step\n")
        #print(global_step)
        #print("model.loss\n")
        #print(model.loss)
        #print("model.optimize\n")
        #print(model.optimize)
        #step, loss, opt, inputs2, mel2, stop2,linear2 = sess.run([global_step, model.loss, model.optimize,tf.shape(model.Inputs2),tf.shape(model.Mel_targets2),tf.shape(model.Stop_token_targets2),tf.shape(model.Linear_targets2)], options=run_options, run_metadata=None)
        #log('inputs2 :{}'.format(inputs2))
        #log('mel2 :{}'.format(mel2))
        #log('stop2 :{}'.format(stop2))
        #log('linear2 :{}'.format(linear2))
          # print(step)
        #coord.request_stop()
        #coord.join(queue_runner)
        #log("ok")
        step, loss, opt = sess.run([global_step, model.loss, model.optimize], options=run_options, run_metadata=run_metadata)    
        time_window.append(time.time() - start_time)
        loss_window.append(loss)
        #if step >= args.build_cost_model_after:   
        #    message = 'Step {:7d} [{:.3f} sec/step, loss={:.5f}, avg_loss={:.5f}]'.format(
        #    step, time_window.average, loss, loss_window.average)
        #    log(message, end='\n', slack=(step % args.checkpoint_interval == 0))
        message = 'Step {:7d} [{:.3f} sec/step, loss={:.5f}, avg_loss={:.5f}]'.format(
        step, time_window.average, loss, loss_window.average)
        if step > 10:
          speed += time_window.average
        log(message, end='\n', slack=(step % args.checkpoint_interval == 0))
        
        if run_metadata != None:
          mem_draw_util.get_alloc_infos(net=args.model,
                                        batch_size=hparams.tacotron_batch_size,
                                        step_id=step,
                                        is_train=True,
                                        run_metadata=run_metadata,
                                        draw=False)

        # if np.isnan(loss) or loss > 100.:
        #   log('Loss exploded to {:.5f} at step {}'.format(loss, step))
        #   raise Exception('Loss exploded')

        if step % args.summary_interval == 0:
          log('\nWriting summary at step {}'.format(step))
          summary_writer.add_summary(sess.run(stats), step)

        if step % args.eval_interval == 0:
          #Run eval and save eval stats
          log('\nRunning evaluation at step {}'.format(step))

          eval_losses = []
          before_losses = []
          after_losses = []
          stop_token_losses = []
          linear_losses = []
          linear_loss = None

          if hparams.predict_linear:
            for i in tqdm(range(feeder.test_steps)):
              eloss, before_loss, after_loss, stop_token_loss, linear_loss, mel_p, mel_t, t_len, align, lin_p, lin_t = sess.run([
                eval_model.tower_loss[0], eval_model.tower_before_loss[0], eval_model.tower_after_loss[0],
                eval_model.tower_stop_token_loss[0], eval_model.tower_linear_loss[0], eval_model.tower_mel_outputs[0][0],
                eval_model.tower_mel_targets[0][0], eval_model.tower_targets_lengths[0][0],
                eval_model.tower_alignments[0][0], eval_model.tower_linear_outputs[0][0],
                eval_model.tower_linear_targets[0][0],
                ])
              eval_losses.append(eloss)
              before_losses.append(before_loss)
              after_losses.append(after_loss)
              stop_token_losses.append(stop_token_loss)
              linear_losses.append(linear_loss)
            linear_loss = sum(linear_losses) / len(linear_losses)

            if hparams.GL_on_GPU:
              wav = sess.run(GLGPU_lin_outputs, feed_dict={GLGPU_lin_inputs: lin_p})
              wav = audio.inv_preemphasis(wav, hparams.preemphasis, hparams.preemphasize)
            else:
              wav = audio.inv_linear_spectrogram(lin_p.T, hparams)
            audio.save_wav(wav, os.path.join(eval_wav_dir, 'step-{}-eval-wave-from-linear.wav'.format(step)), sr=hparams.sample_rate)

          else:
            for i in tqdm(range(feeder.test_steps)):
              eloss, before_loss, after_loss, stop_token_loss, mel_p, mel_t, t_len, align = sess.run([
                eval_model.tower_loss[0], eval_model.tower_before_loss[0], eval_model.tower_after_loss[0],
                eval_model.tower_stop_token_loss[0], eval_model.tower_mel_outputs[0][0], eval_model.tower_mel_targets[0][0],
                eval_model.tower_targets_lengths[0][0], eval_model.tower_alignments[0][0]
                ])
              eval_losses.append(eloss)
              before_losses.append(before_loss)
              after_losses.append(after_loss)
              stop_token_losses.append(stop_token_loss)

          eval_loss = sum(eval_losses) / len(eval_losses)
          before_loss = sum(before_losses) / len(before_losses)
          after_loss = sum(after_losses) / len(after_losses)
          stop_token_loss = sum(stop_token_losses) / len(stop_token_losses)

          log('Saving eval log to {}..'.format(eval_dir))
          #Save some log to monitor model improvement on same unseen sequence
          if hparams.GL_on_GPU:
            wav = sess.run(GLGPU_mel_outputs, feed_dict={GLGPU_mel_inputs: mel_p})
            wav = audio.inv_preemphasis(wav, hparams.preemphasis, hparams.preemphasize)
          else:
            wav = audio.inv_mel_spectrogram(mel_p.T, hparams)
          audio.save_wav(wav, os.path.join(eval_wav_dir, 'step-{}-eval-wave-from-mel.wav'.format(step)), sr=hparams.sample_rate)

          plot.plot_alignment(align, os.path.join(eval_plot_dir, 'step-{}-eval-align.png'.format(step)),
            title='{}, {}, step={}, loss={:.5f}'.format(args.model, time_string(), step, eval_loss),
            max_len=t_len // hparams.outputs_per_step)
          plot.plot_spectrogram(mel_p, os.path.join(eval_plot_dir, 'step-{}-eval-mel-spectrogram.png'.format(step)),
            title='{}, {}, step={}, loss={:.5f}'.format(args.model, time_string(), step, eval_loss), target_spectrogram=mel_t,
            max_len=t_len)

          if hparams.predict_linear:
            plot.plot_spectrogram(lin_p, os.path.join(eval_plot_dir, 'step-{}-eval-linear-spectrogram.png'.format(step)),
              title='{}, {}, step={}, loss={:.5f}'.format(args.model, time_string(), step, eval_loss), target_spectrogram=lin_t,
              max_len=t_len, auto_aspect=True)

          log('Eval loss for global step {}: {:.3f}'.format(step, eval_loss))
          log('Writing eval summary!')
          add_eval_stats(summary_writer, step, linear_loss, before_loss, after_loss, stop_token_loss, eval_loss)


        if step % args.checkpoint_interval == 0 or step == args.tacotron_train_steps or step == 300:
          #Save model and current global step
          saver.save(sess, checkpoint_path, global_step=global_step)

          log('\nSaving alignment, Mel-Spectrograms and griffin-lim inverted waveform..')
          if hparams.predict_linear:
            input_seq, mel_prediction, linear_prediction, alignment, target, target_length, linear_target = sess.run([
              model.tower_inputs[0][0],
              model.tower_mel_outputs[0][0],
              model.tower_linear_outputs[0][0],
              model.tower_alignments[0][0],
              model.tower_mel_targets[0][0],
              model.tower_targets_lengths[0][0],
              model.tower_linear_targets[0][0],
              ])

            #save predicted linear spectrogram to disk (debug)
            linear_filename = 'linear-prediction-step-{}.npy'.format(step)
            np.save(os.path.join(linear_dir, linear_filename), linear_prediction.T, allow_pickle=False)

            #save griffin lim inverted wav for debug (linear -> wav)
            if hparams.GL_on_GPU:
              wav = sess.run(GLGPU_lin_outputs, feed_dict={GLGPU_lin_inputs: linear_prediction})
              wav = audio.inv_preemphasis(wav, hparams.preemphasis, hparams.preemphasize)
            else:
              wav = audio.inv_linear_spectrogram(linear_prediction.T, hparams)
            audio.save_wav(wav, os.path.join(wav_dir, 'step-{}-wave-from-linear.wav'.format(step)), sr=hparams.sample_rate)

            #Save real and predicted linear-spectrogram plot to disk (control purposes)
            plot.plot_spectrogram(linear_prediction, os.path.join(plot_dir, 'step-{}-linear-spectrogram.png'.format(step)),
              title='{}, {}, step={}, loss={:.5f}'.format(args.model, time_string(), step, loss), target_spectrogram=linear_target,
              max_len=target_length, auto_aspect=True)

          else:
            input_seq, mel_prediction, alignment, target, target_length = sess.run([
              model.tower_inputs[0][0],
              model.tower_mel_outputs[0][0],
              model.tower_alignments[0][0],
              model.tower_mel_targets[0][0],
              model.tower_targets_lengths[0][0],
              ])

          #save predicted mel spectrogram to disk (debug)
          mel_filename = 'mel-prediction-step-{}.npy'.format(step)
          np.save(os.path.join(mel_dir, mel_filename), mel_prediction.T, allow_pickle=False)

          #save griffin lim inverted wav for debug (mel -> wav)
          if hparams.GL_on_GPU:
            wav = sess.run(GLGPU_mel_outputs, feed_dict={GLGPU_mel_inputs: mel_prediction})
            wav = audio.inv_preemphasis(wav, hparams.preemphasis, hparams.preemphasize)
          else:
            wav = audio.inv_mel_spectrogram(mel_prediction.T, hparams)
          audio.save_wav(wav, os.path.join(wav_dir, 'step-{}-wave-from-mel.wav'.format(step)), sr=hparams.sample_rate)

          #save alignment plot to disk (control purposes)
          plot.plot_alignment(alignment, os.path.join(plot_dir, 'step-{}-align.png'.format(step)),
            title='{}, {}, step={}, loss={:.5f}'.format(args.model, time_string(), step, loss),
            max_len=target_length // hparams.outputs_per_step)
          #save real and predicted mel-spectrogram plot to disk (control purposes)
          plot.plot_spectrogram(mel_prediction, os.path.join(plot_dir, 'step-{}-mel-spectrogram.png'.format(step)),
            title='{}, {}, step={}, loss={:.5f}'.format(args.model, time_string(), step, loss), target_spectrogram=target,
            max_len=target_length)
          log('Input at step {}: {}'.format(step, sequence_to_text(input_seq)))

        if step % args.embedding_interval == 0 or step == args.tacotron_train_steps or step == 1:
          #Get current checkpoint state
          checkpoint_state = tf.train.get_checkpoint_state(save_dir)

          #Update Projector
          log('\nSaving Model Character Embeddings visualization..')
          add_embedding_stats(summary_writer, [model.embedding_table.name], [char_embedding_meta], checkpoint_state.model_checkpoint_path)
          log('Tacotron Character embeddings have been updated on tensorboard!')

      log('Tacotron training complete after {} global steps!'.format(args.tacotron_train_steps), slack=True)
      log('speed: {}'.format(speed / (args.tacotron_train_steps - 10)))
      return save_dir

    except Exception as e:
      log('Exiting due to exception: {}'.format(e), slack=True)
      traceback.print_exc()
      coord.request_stop(e)

def tacotron_train(args, log_dir, hparams):
  return train(log_dir, args, hparams)
