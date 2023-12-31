import numpy as np
import random
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

critical_area_agent = 0.2
critical_area_obstacle = 0.2
target_range = 0.15

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 2
        num_landmarks = 2
        num_obs = 2
        num_uneven = 0
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.05
            agent.accel = 3.0
            agent.max_speed = 1.0
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        world.landmarks_reached = [False for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.1
        # add obstacles
        world.obs = [Landmark() for i in range(num_obs)]
        for i, landmark in enumerate(world.obs):
            landmark.name = 'obs %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.15
            landmark.boundary = False
        # add rough terrain
        world.uneven = [Landmark() for i in range(num_uneven)]
        for i, landmark in enumerate(world.uneven):
            landmark.name = 'uneven %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.15
            landmark.boundary = False
        world.landmarks += world.obs
        world.landmarks += world.uneven
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        for i, landmark in enumerate(world.obs):
            landmark.color = np.array([0.15, 0.15, 0.65])
        for i, landmark in enumerate(world.uneven):
            landmark.color = np.array([0.6, 0.9, 0.6])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        for i, landmark in enumerate(world.obs):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        for i, landmark in enumerate(world.uneven):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        for i in range(len(world.landmarks_reached)):
            world.landmarks_reached[i] = False

    def benchmark_data(self, agent, world):
        rew = 0
        ag_collisions = 0
        ob_collisions = 0
        un_collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        i = 0
        for l in world.landmarks:
            if i == len(world.agents):
                break
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) <= target_range:
                occupied_landmarks += 1
                world.landmarks_reached[i] = True
            i += 1

        '''for o in world.obs:
            delta_pos = agent1.state.p_pos - agent2.state.p_pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            dist_min = agent1.size + agent2.size
            if dist > dist_min:
                dist = dist - dist_min
                if dist < 0.1:
                    rew -= ((0.1 - dist) / 0.1)
            else:
                rew -= 1
                ob_collisions += 1'''

        if agent.collide:
            for a in world.agents:
                if agent != a:
                    delta_pos = agent.state.p_pos - a.state.p_pos
                    dist = np.sqrt(np.sum(np.square(delta_pos)))
                    dist_min = agent.size + a.size
                    if dist < dist_min + critical_area_agent:
                        # rew -= ((dist_min + critical_area_agent - dist) / critical_area_agent)
                        rew -= ((dist_min + critical_area_agent - dist))
                    if self.is_collision(agent, a):
                        ag_collisions += 1
            for o in world.obs:
                delta_pos = agent.state.p_pos - o.state.p_pos
                dist = np.sqrt(np.sum(np.square(delta_pos)))
                dist_min = agent.size + o.size
                if dist < dist_min + critical_area_obstacle:
                    #rew -= ((dist_min + critical_area_obstacle - dist) / critical_area_obstacle)
                    rew -= ((dist_min + critical_area_obstacle - dist))
                if self.is_collision(agent, o):
                    ob_collisions += 1
            for u in world.uneven:
                delta_pos = agent.state.p_pos - u.state.p_pos
                dist = np.sqrt(np.sum(np.square(delta_pos)))
                dist_min = agent.size + u.size
                if dist < dist_min + critical_area_obstacle:
                    #rew -= 0.25*((dist_min + critical_area_obstacle - dist) / critical_area_obstacle)
                    rew -= 0.1*((dist_min + critical_area_obstacle - dist))
                if self.is_collision(agent, u):
                    un_collisions += 1
        return (rew, ag_collisions, ob_collisions, un_collisions, min_dists, sum(world.landmarks_reached))

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        i = 0
        for l in world.landmarks:
            if i == len(world.agents):
                break
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            rew -= min(dists)
            i += 1
        if agent.collide:
            for a in world.agents:
                if agent != a:
                    delta_pos = agent.state.p_pos - a.state.p_pos
                    dist = np.sqrt(np.sum(np.square(delta_pos)))
                    dist_min = agent.size + a.size
                    if dist < dist_min + critical_area_agent:
                        #rew -= ((dist_min + critical_area_agent - dist) / critical_area_agent)
                        rew -= ((dist_min + critical_area_agent - dist))
            for o in world.obs:
                delta_pos = agent.state.p_pos - o.state.p_pos
                dist = np.sqrt(np.sum(np.square(delta_pos)))
                dist_min = agent.size + o.size
                if dist < dist_min + critical_area_obstacle:
                    #rew -= ((dist_min + critical_area_obstacle - dist) / critical_area_obstacle)
                    rew -= ((dist_min + critical_area_obstacle - dist))
            for u in world.uneven:
                delta_pos = agent.state.p_pos - u.state.p_pos
                dist = np.sqrt(np.sum(np.square(delta_pos)))
                dist_min = agent.size + u.size
                if dist < dist_min + critical_area_obstacle:
                    #rew -= 0.25*((dist_min + critical_area_obstacle - dist) / critical_area_obstacle)
                    rew -= 0.1*((dist_min + critical_area_obstacle - dist))
            return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        '''#position of obstacles
        obs_pos = []
        for entity in world.obs:  # world.entities:
            obs_pos.append(entity.state.p_pos - agent.state.p_pos)
        #position of obstacles
        uneven_pos = []
        for entity in world.uneven:  # world.entities:
            uneven_pos.append(entity.state.p_pos - agent.state.p_pos)'''
        # position of other agents
        other_ag_pos = []
        for other in world.agents:
            if other is agent: continue
            other_ag_pos.append(other.state.p_pos - agent.state.p_pos)
        # return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_ag_pos + obs_pos + uneven_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_ag_pos)
