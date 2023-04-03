import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PLARMixer(nn.Module):
    def __init__(self, args):
        super(PLARMixer, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions

        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))

            self.mf_hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_actions))


            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                               nn.ReLU(),
                                               nn.Linear(hypernet_embed, self.embed_dim))

            self.mf_hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                               nn.ReLU(),
                                               nn.Linear(hypernet_embed, self.embed_dim))




        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")


        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        self.mf_hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states, mf_actions):
        bs = agent_qs.size(0)

        mf_actions = mf_actions.reshape(-1, 1, self.n_actions)

        states = states.reshape(-1, self.state_dim)


        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)


        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)



        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)

        mf_w1 = th.abs(self.mf_hyper_w_1(states))
        mf_w1 = mf_w1.view(-1, self.n_actions, self.embed_dim)

        mf_b1 = self.mf_hyper_b_1(states)
        mf_b1 = mf_b1.view(-1, 1, self.embed_dim)

        mf_w_final = th.abs(self.mf_hyper_w_final(states))
        mf_w_final = mf_w_final.view(-1, self.embed_dim, 1)


        hidden = F.elu(th.bmm(mf_actions.clone().detach(), mf_w1) + mf_b1)
        delu = th.ones_like(hidden)
        mask = hidden < 0
        delu[mask] = th.exp(hidden[mask])
        action_honor_value = th.bmm(mf_w1, th.diag_embed(delu.squeeze(1)))
        action_honor_value = th.bmm(action_honor_value, mf_w_final)
        action_honor_value = action_honor_value.reshape(bs, -1, self.n_actions)

        return q_tot,action_honor_value
