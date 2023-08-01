from copy import deepcopy

import numpy as np
import random
import json
import math
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

critical_area_agent = 0.2
critical_area_obstacle = 0.2
target_range = 0.15

local_dn_fg = []
UGV_rew = 1
ep_cnt = -2
UGV_pos = []
UAV_dn_dis = []

class Scenario(BaseScenario):
    def make_world(self):
        global UAV_dn_dis
        global local_dn_fg
        world = World()

        #with open('run1/F_ARC_ORC_3UGV_data_ep_150000.json', 'r') as openfile:
            # Reading from json file
         #   dict = json.load(openfile)
         #   UGV_pos = dict['pos']
        # set any world properties first
        for i in range(150000):
            temp = []
            temp.append(np.random.uniform(-1, +1, world.dim_p))
            temp.append(np.random.uniform(-1, +1, world.dim_p))
            UGV_pos.append(deepcopy(temp))

        world.dim_c = 2
        num_agents = 2
        num_landmarks = 2
        num_obs = 2
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
        world.landmarks_avail = [True for i in range(num_landmarks)]
        local_dn_fg = [False for i in range(num_agents)]
        UAV_dn_dis = [math.inf for i in range(num_agents)]
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
        world.landmarks += world.obs
        # make initial conditions
        world.UGV = [Landmark() for i in range(1)]
        for i, landmark in enumerate(world.UGV):
            landmark.name = 'UGV %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.boundary = False

        world.landmarks += world.UGV
        self.reset_world(world)

        return world

    def reset_world(self, world):
        global ep_cnt
        global UGV_pos
        global UAV_dn_dis
        global local_dn_fg
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        for i, landmark in enumerate(world.obs):
            landmark.color = np.array([0.15, 0.15, 0.65])
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

        for i in range(len(world.landmarks_reached)):
            world.landmarks_reached[i] = False
            world.landmarks_avail[i] = True
        for i in range(len(local_dn_fg)):
            local_dn_fg[i] = False
        for i in range(len(UAV_dn_dis)):
            UAV_dn_dis[i] = math.inf
        ep_cnt += 1
        if ep_cnt == -1:
            for agent in world.agents:
                agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
        elif ep_cnt > -1:
            delta = 0
            for agent in world.agents:
                #print(agent.state.p_pos)
                agent.state.p_pos[0] = UGV_pos[ep_cnt][0][0]
                agent.state.p_pos[1] = UGV_pos[ep_cnt][0][1] + delta
                delta += world.agents[0].size*2
            for i, landmark in enumerate(world.UGV):
                landmark.state.p_pos[0] = UGV_pos[ep_cnt][-1][0]
                landmark.state.p_pos[1] = UGV_pos[ep_cnt][-1][1]
                landmark.state.p_vel = np.zeros(world.dim_p)



    def benchmark_data(self, agent, world):
        global UAV_dn_dis
        global local_dn_fg
        global ep_cnt
        global UGV_rew
        global UGV_pos
        rew = 0
        ag_collisions = 0
        ob_collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        idx = -1
        cls_lm = -1
        for i, a in enumerate(world.agents):
            if agent == a:
                idx = i


        if not local_dn_fg[idx]:
            min_dists = math.inf
            i = 0
            for l in world.landmarks:
                if i == len(world.agents):
                    break
                if world.landmarks_avail[i]:
                    dists = np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos)))
                    if dists < min_dists:
                        min_dists = dists
                        cls_lm = i
                i += 1
            if min_dists <= target_range:
                local_dn_fg[idx] = True
                world.landmarks_avail[cls_lm] = False
                UAV_dn_dis[idx] = np.sqrt(np.sum(np.square(agent.state.p_pos - UGV_pos[ep_cnt][-1])))

        min_dists = 0
        if not local_dn_fg[idx]:
            i = 0
            for l in world.landmarks:
                if i == len(world.agents):
                    break
                if world.landmarks_avail[i]:
                    dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
                    min_dists += min(dists)
                    rew -= min(dists)
                i += 1
                    #world.landmarks_reached[i] = True'''

        if agent.collide:
            for a in world.agents:
                if agent != a:
                    delta_pos = agent.state.p_pos - a.state.p_pos
                    dist = np.sqrt(np.sum(np.square(delta_pos)))
                    dist_min = agent.size + a.size
                    if dist < dist_min + critical_area_agent:
                        #rew -= ((dist_min + critical_area_agent - dist) / critical_area_agent)
                        rew -= (dist_min + critical_area_agent - dist)# / critical_area_agent)
                    if self.is_collision(agent, a):
                        ag_collisions += 1
            for o in world.obs:
                delta_pos = agent.state.p_pos - o.state.p_pos
                dist = np.sqrt(np.sum(np.square(delta_pos)))
                dist_min = agent.size + o.size
                if dist < dist_min + critical_area_obstacle:
                    #rew -= ((dist_min + critical_area_obstacle - dist) / critical_area_obstacle)
                    rew -= (dist_min + critical_area_obstacle - dist) #/ critical_area_obstacle)
                if self.is_collision(agent, o):
                    ob_collisions += 1
        if not local_dn_fg[idx]:
            rew -= 0#UGV_rew
        else:
            #if not world.landmarks_reached[idx]:
            delta_pos = agent.state.p_pos - world.UGV[0].state.p_pos# UGV_pos[ep_cnt][-1]
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            rew -= dist
            if dist <= target_range:
                world.landmarks_reached[idx] = True

            #else:
                #rew -= UGV_rew * (dist/UAV_dn_dis[idx])
                #rew -= dist#UGV_rew * (dist/UAV_dn_dis[idx])


        return (rew, ag_collisions, ob_collisions, min_dists, sum(world.landmarks_reached))

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        global UAV_dn_dis
        global local_dn_fg
        global ep_cnt
        global UGV_rew
        global UGV_pos
        rew = 0
        ag_collisions = 0
        ob_collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        idx = -1
        cls_lm = -1
        for i, a in enumerate(world.agents):
            if agent == a:
                idx = i

        if not local_dn_fg[idx]:
            min_dists = math.inf
            i = 0
            for l in world.landmarks:
                if i == len(world.agents):
                    break
                if world.landmarks_avail[i]:
                    dists = np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos)))
                    if dists < min_dists:
                        min_dists = dists
                        cls_lm = i
                i += 1
            if min_dists <= target_range:
                local_dn_fg[idx] = True
                world.landmarks_avail[cls_lm] = False
                UAV_dn_dis[idx] = np.sqrt(np.sum(np.square(agent.state.p_pos - UGV_pos[ep_cnt][-1])))

        min_dists = 0
        if not local_dn_fg[idx]:
            i = 0
            for l in world.landmarks:
                if i == len(world.agents):
                    break
                if world.landmarks_avail[i]:
                    dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
                    min_dists += min(dists)
                    rew -= min(dists)
                i += 1
                # world.landmarks_reached[i] = True
        if agent.collide:
            for a in world.agents:
                if agent != a:
                    delta_pos = agent.state.p_pos - a.state.p_pos
                    dist = np.sqrt(np.sum(np.square(delta_pos)))
                    dist_min = agent.size + a.size
                    if dist < dist_min + critical_area_agent:
                        # rew -= ((dist_min + critical_area_agent - dist) / critical_area_agent)
                        rew -= (dist_min + critical_area_agent - dist)  # / critical_area_agent)
                    if self.is_collision(agent, a):
                        ag_collisions += 1
            for o in world.obs:
                delta_pos = agent.state.p_pos - o.state.p_pos
                dist = np.sqrt(np.sum(np.square(delta_pos)))
                dist_min = agent.size + o.size
                if dist < dist_min + critical_area_obstacle:
                    # rew -= ((dist_min + critical_area_obstacle - dist) / critical_area_obstacle)
                    rew -= (dist_min + critical_area_obstacle - dist)  # / critical_area_obstacle)
                if self.is_collision(agent, o):
                    ob_collisions += 1
        if not local_dn_fg[idx]:
            rew -= 0#UGV_rew
        else:
            # if not world.landmarks_reached[idx]:
            delta_pos = agent.state.p_pos - world.UGV[0].state.p_pos  # UGV_pos[ep_cnt][-1]
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            rew -= dist
            if dist <= target_range:
                world.landmarks_reached[idx] = True

            # else:
            # rew -= UGV_rew * (dist/UAV_dn_dis[idx])
            # rew -= dist#UGV_rew * (dist/UAV_dn_dis[idx])

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
        x = int(sum(world.landmarks_reached) / len(world.landmarks_reached))
        # return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_ag_pos + obs_pos + uneven_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_ag_pos)
