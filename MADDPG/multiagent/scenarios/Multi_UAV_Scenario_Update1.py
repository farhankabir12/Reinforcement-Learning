import numpy as np
import random
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
critical_area_agent = 0.2
critical_area_obstacle = 0.2
target_range = 0.15
UGV_range = 0.15

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 2
        num_landmarks = 2
        num_obs = 2
        num_UGVs = 1
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.075
            agent.accel = 3.0
            agent.max_speed = 1.0
        # flag for UAV to verify if it reached target
        world.agent_touched_lm = [False for i in range(num_agents)]
        world.agent_UGV_max_dis = [1 for i in range(num_agents)]
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        world.landmark_reached = [False for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # add obstacles
        world.obs = [Landmark() for i in range(num_obs)]
        for i, landmark in enumerate(world.obs):
            landmark.name = 'obs %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.1
            landmark.boundary = False
        world.landmarks += world.obs
        # add UGV positions
        world.UGVs = [Landmark() for i in range(num_UGVs)]
        for i, landmark in enumerate(world.UGVs):
            landmark.name = 'UGV %d' % i
            landmark.collide = False
            landmark.movable = False
        world.landmarks += world.UGVs
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
        for i, landmark in enumerate(world.UGVs):
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
            landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        for i, landmark in enumerate(world.UGVs):
            landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        for i in range(len(world.landmark_reached)):
            world.landmark_reached[i] = False
        for i in range(len(world.agent_touched_lm)):
            world.agent_touched_lm[i] = False
        for i in range(len(world.agent_UGV_max_dis)):
            world.agent_UGV_max_dis[i] = 1

    def benchmark_data(self, agent, world):
        rew = 0
        ag_collisions = 0
        ob_collisions = 0
        idx = -1
        if agent.collide:
            for a in world.agents:
                if agent != a:
                    delta_pos = agent.state.p_pos - a.state.p_pos
                    dist = np.sqrt(np.sum(np.square(delta_pos)))
                    dist_min = agent.size + a.size
                    if dist < dist_min + critical_area_agent:
                        rew -= ((dist_min + critical_area_agent - dist)/critical_area_agent) # scaled reward
                        #rew -= ((dist_min + critical_area_agent - dist)) # non-scaled reward
                    if self.is_collision(agent, a):
                        ag_collisions += 1
            for o in world.obs:
                delta_pos = agent.state.p_pos - o.state.p_pos
                dist = np.sqrt(np.sum(np.square(delta_pos)))
                dist_min = agent.size + o.size
                if dist < dist_min + critical_area_obstacle:
                    rew -= ((dist_min + critical_area_obstacle - dist)/critical_area_obstacle) # scaled reward
                    #rew -= ((dist_min + critical_area_obstacle - dist)) # non-scaled reward
                if self.is_collision(agent, o):
                    ob_collisions += 1
        for i, a in enumerate(world.agents):
            if a == agent:
                idx = i
                break
        if world.agent_touched_lm[idx]:
            dists = [np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos))) for l in world.UGVs]
            min_dist = min(dists)
            if min_dist > UGV_range:
                rew -= min_dist/world.agent_UGV_max_dis[idx] # scaled reward
                #rew -= min_dist # non-scaled reward
        else:
            rew -= 1
            dists = []
            min_dist = 0
            for i, l in enumerate(world.landmarks):
                if i == len(world.agents):
                    break
                if not world.landmark_reached[i]:
                    delta_pos = agent.state.p_pos - l.state.p_pos
                    dist = np.sqrt(np.sum(np.square(delta_pos)))
                    dists.append(dist)
                    if dist < target_range:
                        #world.landmark_reached[i] = True            # for reward function only
                        #world.agent_touched_lm[idx] = True          # for reward function only
                        # get distance of closest UGV from landmark, here we only used 1 UGV
                        delta_pos_UGV = world.UGVs[0].state.p_pos - l.state.p_pos
                        #world.agent_UGV_max_dis[idx] = np.sqrt(np.sum(np.square(delta_pos_UGV)))
                        break
            if not world.agent_touched_lm[idx]:
                min_dist += min(dists)
                for i, l in enumerate(world.landmarks):
                    dists = []
                    if i == len(world.agents):
                        break
                    if world.landmark_reached[i]:
                        continue
                    for j, a in enumerate(world.agents):
                        if a != agent:
                            if not world.agent_touched_lm[j]:
                                delta_pos = a.state.p_pos - l.state.p_pos
                                dist = np.sqrt(np.sum(np.square(delta_pos)))
                                dists.append(dist)
                    if len(dists):
                        min_dist += min(dists)
                rew -= min_dist
        
        UGVReached = 0
        if sum(world.agent_touched_lm) == len(world.agents):
            for a in world.agents:
                dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for l in world.UGVs]
                min_dist = min(dists)
                if min_dist < UGV_range:
                    UGVReached += 1
        return (rew, ag_collisions, ob_collisions, 0, 0, UGVReached)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        ag_collisions = 0
        ob_collisions = 0
        idx = -1
        if agent.collide:
            for a in world.agents:
                if agent != a:
                    delta_pos = agent.state.p_pos - a.state.p_pos
                    dist = np.sqrt(np.sum(np.square(delta_pos)))
                    dist_min = agent.size + a.size
                    if dist < dist_min + critical_area_agent:
                        rew -= ((dist_min + critical_area_agent - dist)/critical_area_agent) # scaled reward
                        #rew -= ((dist_min + critical_area_agent - dist)) # non-scaled reward
                    if self.is_collision(agent, a):
                        ag_collisions += 1
            for o in world.obs:
                delta_pos = agent.state.p_pos - o.state.p_pos
                dist = np.sqrt(np.sum(np.square(delta_pos)))
                dist_min = agent.size + o.size
                if dist < dist_min + critical_area_obstacle:
                    rew -= ((dist_min + critical_area_obstacle - dist)/critical_area_obstacle) # scaled reward
                    #rew -= ((dist_min + critical_area_obstacle - dist)) # non-scaled reward
                if self.is_collision(agent, o):
                    ob_collisions += 1
        for i, a in enumerate(world.agents):
            if a == agent:
                idx = i
                break
        if world.agent_touched_lm[idx]:
            dists = [np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos))) for l in world.UGVs]
            min_dist = min(dists)
            if min_dist > UGV_range:
                rew -= min_dist/world.agent_UGV_max_dis[idx] # scaled reward
                #rew -= min_dist # non-scaled reward
        else:
            rew -= 1
            dists = []
            min_dist = 0
            for i, l in enumerate(world.landmarks):
                if i == len(world.agents):
                    break
                if not world.landmark_reached[i]:
                    delta_pos = agent.state.p_pos - l.state.p_pos
                    dist = np.sqrt(np.sum(np.square(delta_pos)))
                    dists.append(dist)
                    if dist < target_range:
                        world.landmark_reached[i] = True            # for reward function only
                        world.agent_touched_lm[idx] = True          # for reward function only
                        # get distance of closest UGV from landmark, here we only used 1 UGV
                        delta_pos_UGV = world.UGVs[0].state.p_pos - l.state.p_pos
                        world.agent_UGV_max_dis[idx] = np.sqrt(np.sum(np.square(delta_pos_UGV)))
                        break
            if not world.agent_touched_lm[idx]:
                min_dist += min(dists)
                for i, l in enumerate(world.landmarks):
                    dists = []
                    if i == len(world.agents):
                        break
                    if world.landmark_reached[i]:
                        continue
                    for j, a in enumerate(world.agents):
                        if a != agent:
                            if not world.agent_touched_lm[j]:
                                delta_pos = a.state.p_pos - l.state.p_pos
                                dist = np.sqrt(np.sum(np.square(delta_pos)))
                                dists.append(dist)
                    if len(dists):
                        min_dist += min(dists)
                rew -= min_dist
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # position of other agents
        idx = -1
        other_ag_pos = []
        for i, other in enumerate(world.agents):
            if other is agent:
                idx = i
            else:
                other_ag_pos.append(other.state.p_pos - agent.state.p_pos)
        ag_lm_state = []
        if world.agent_touched_lm[idx]:
            ag_lm_state.append(0)
        else:
            ag_lm_state.append(1) 
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_ag_pos + [world.agent_touched_lm] + [world.landmark_reached])
