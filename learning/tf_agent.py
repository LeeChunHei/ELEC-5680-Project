import numpy as np
import torch
from abc import abstractmethod

from learning.rl_agent import RLAgent
from learning.normalizer import Normalizer
from learning.tf_normalizer import TFNormalizer

class TFAgent(RLAgent):
    RESOURCE_SCOPE = 'resource'
    SOLVER_SCOPE = 'solvers'

    def __init__(self, json_data, env, writer):
        self.models_name = []    #for save model, store [attribute name, type]

        super().__init__(json_data, env, writer)
        self._build_graph(json_data)
        self._init_normalizers()
        return

    def __del__(self):
        return

    def save_model(self, out_path):
        pass
        #TODO: need to see what to save, and need to be called by upper class
        # assert(False)
        # with self.sess.as_default(), self.graph.as_default():
        #     try:
        #         save_path = self.saver.save(self.sess, out_path, write_meta_graph=False, write_state=False)
        #         Logger.print('Model saved to: ' + save_path)
        #     except:
        #         Logger.print("Failed to save model to: " + save_path)
        # return

    def load_model(self, in_path):
        pass
        # assert(False)
        # with self.sess.as_default(), self.graph.as_default():
        #     self.saver.restore(self.sess, in_path)
        #     self._load_normalizers()
        #     Logger.print('Model loaded from: ' + in_path)
        # return

    def _get_output_path(self):
        assert(self.output_dir != '')
        file_path = self.output_dir + '/agent' + str(0) + '_model.ckpt'
        return file_path

    def _get_int_output_path(self):
        assert(self.int_output_dir != '')
        file_path = self.int_output_dir + ('/agent{:d}_models/agent{:d}_int_model_{:010d}.ckpt').format(0, 0, self.iter)
        return file_path

    def _build_graph(self, json_data):
        self._build_nets(json_data)
        self._build_losses(json_data)
        self._build_solvers(json_data)
        self._initialize_vars()
        self._build_saver()
        return

    def _init_normalizers(self):
        self._s_norm.update()
        self._g_norm.update()
        self._a_norm.update()
        return

    @abstractmethod
    def _build_nets(self, json_data):
        pass

    @abstractmethod
    def _build_losses(self, json_data):
        pass

    @abstractmethod
    def _build_solvers(self, json_data):
        pass

    def _build_normalizers(self):
        self._s_norm = TFNormalizer(self.get_state_size(), self.env.build_state_norm_groups(0))
        self._s_norm.set_mean_std(-self.env.build_state_offset(0), 
                                    1 / self.env.build_state_scale(0))
        
        self._g_norm = TFNormalizer(self.get_goal_size(), self.env.build_goal_norm_groups(0))
        self._g_norm.set_mean_std(-self.env.build_goal_offset(0), 
                                    1 / self.env.build_goal_scale(0))

        self._a_norm = TFNormalizer(self.get_action_size())
        self._a_norm.set_mean_std(-self.env.build_action_offset(0), 
                                    1 / self.env.build_action_scale(0))
        return

    def _load_normalizers(self):
        #TODO: not sure if load can be comment in Normalizer
        self._s_norm.load()
        self._g_norm.load()
        self._a_norm.load()
        return

    def _update_normalizers(self):
        super()._update_normalizers()
        return

    def _initialize_vars(self):
        # self.sess.run(tf.global_variables_initializer())
        return

    def _build_saver(self):
        # vars = self._get_saver_vars()
        # self.saver = tf.train.Saver(vars, max_to_keep=0)
        return

    # def _get_saver_vars(self):
    #     with self.sess.as_default(), self.graph.as_default():
    #         vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.tf_scope)
    #         vars = [v for v in vars if '/' + self.SOLVER_SCOPE + '/' not in v.name]
    #         #vars = [v for v in vars if '/target/' not in v.name]
    #         assert len(vars) > 0
    #     return vars
    
    def _weight_decay_loss(self, scope):
        vars = self._tf_vars(scope)
        vars_no_bias = [v for v in vars if 'bias' not in v.name]
        loss = tf.add_n([tf.nn.l2_loss(v) for v in vars_no_bias])
        return loss

    def _train(self):
        super()._train()
        return