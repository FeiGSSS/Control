import numpy as np
import networkx as nx

class SpringSim(object):
    def __init__(self, n_balls=5, box_size=5., loc_std=3, vel_norm=.5, interaction_strength=.1, noise_var=0.):
        self.n_balls = n_balls
        self.box_size = box_size
        self.loc_std = loc_std
        self.vel_norm = vel_norm
        self.interaction_strength = interaction_strength
        self.noise_var = noise_var

        self._delta_T = 0.001
        self._max_F = 0.1 / self._delta_T


    def _clamp(self, loc, vel):
        """
        :param loc: 2xN location at one time stamp
        :param vel: 2xN velocity at one time stamp
        :return: location and velocity after hiting walls and returning after
            elastically colliding with walls
	    clamp all balls in a bounded box
	    """
        assert (np.all(loc < self.box_size * 3))
        assert (np.all(loc > -self.box_size * 3))

        over = loc > self.box_size
        loc[over] = 2 * self.box_size - loc[over]
        assert (np.all(loc <= self.box_size))

        vel[over] = -np.abs(vel[over])

        under = loc < -self.box_size
        loc[under] = -2 * self.box_size - loc[under]
        assert (np.all(loc >= -self.box_size))
        vel[under] = np.abs(vel[under])

        return loc, vel


    def sample_trajectory(self, T=100000, sample_freq=100):
        # n = self.n_balls
        assert (T % sample_freq == 0)
        T_save = int(T / sample_freq)
        counter = 0

        # Sample Graph
        graph = nx.barabasi_albert_graph(n=self.n_balls, m=3)
        adj = nx.adj_matrix(graph).toarray()
        adj = np.tril(adj) + np.tril(adj, -1).T
        np.fill_diagonal(adj, 0)

        # Initialize location and velocity place-holder
        loc = np.zeros((T_save, 2, self.n_balls))
        vel = np.zeros((T_save, 2, self.n_balls))

        loc_next = np.stack(nx.kamada_kawai_layout(graph).values()).T * self.loc_std
        vel_next = np.zeros((2, self.n_balls)) # init v is all zeros

        loc[0, :, :], vel[0, :, :] = self._clamp(loc_next, vel_next)
        counter += 1

        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):
            forces_size = - self.interaction_strength * adj
            np.fill_diagonal(forces_size, 0)  # self forces are zero (fixes division by zero)

            # run leapfrog
            for i in range(1, T):
                F = (forces_size.reshape(1, self.n_balls, self.n_balls) *
                    np.concatenate((
                        np.subtract.outer(loc_next[0, :], loc_next[0, :]).reshape(1, self.n_balls, self.n_balls),
                     	np.subtract.outer(loc_next[1, :], loc_next[1, :]).reshape(1, self.n_balls, self.n_balls)))).sum(axis=-1)
                F[F > self._max_F] = self._max_F
                F[F < -self._max_F] = -self._max_F

                vel_next += self._delta_T * F # 质量都为1，这里直接省去
                loc_next += self._delta_T * vel_next
                loc_next, vel_next = self._clamp(loc_next, vel_next)

                if i % sample_freq == 0:
                    loc[counter, :, :], vel[counter, :, :] = loc_next, vel_next
                    counter += 1

            # Add noise to observations
            loc += np.random.randn(T_save, 2, self.n_balls) * self.noise_var
            vel += np.random.randn(T_save, 2, self.n_balls) * self.noise_var
            return loc, vel, adj
