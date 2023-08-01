import json
import matplotlib.pyplot as plt
import numpy as np


'''dictionary = {
            "task_completion" : list_task_completion,
            "ag_col" : list_ag_col,
            "ob_col" : list_ob_col,
            "un_col" : list_un_col,
            "time_steps" : time_steps,
            "score_history" : score_history,
            "avg_score_history" : avg_score_history
        }'''

# Data to be written
ep_cnt = 150000
num_agents = 2
with open('run1_2UGV/data_ep_' + str(ep_cnt) + '.json', 'r') as openfile:
#with open('C:/Users/Farhan Kabir/Documents/CSE/GNR/LinuxBackups/MAPPO/on-policy-main/tmp/2UGV_Spread/data_ep_' + str(ep_cnt) + '.json', 'r') as openfile:
    # Reading from json file
    dict = json.load(openfile)
    #print(type(json_object['list1'][0]))

    avg_score_history = dict['avg_score_history']
    list_ag_col = dict['ag_col']
    list_ob_col = dict['ob_col']
    list_un_col = dict['un_col']
    list_task_completion = dict['task_completion']
    score_history = dict['score_history']
    time_steps = dict['time_steps']


    ind = []
    AVG_SCORE = []
    AARC = []
    AARC_RATE = []
    ORC = []
    ORC_RATE = []
    URC = []
    URC_RATE = []
    CRT = []
    CTT = []

    crt_sum = 0
    ctt_sum = 0
    avg_score_sum = 0
    aarc_sum = 0
    time_sum = 0

    for i in range(ep_cnt):

        if i >= 1000:
            crt_sum -= list_task_completion[i - 1000]
            ctt_sum -= time_steps[i - 1000]
            avg_score_sum -= score_history[i - 1000]

        ind.append((i + 1))
        avg_score_sum += score_history[i]
        if i + 1 >= 1000:
            AVG_SCORE.append(avg_score_sum / 1000)
        else:
            AVG_SCORE.append(avg_score_sum / (i + 1))
        #aarc_sum += list_ag_col[i]
        #time_sum += (time_steps[i] * num_agents)
        #AARC_RATE.append(aarc_sum / time_sum)
        AARC.append(list_ag_col[i])#/(time_steps[i] * num_agents))
        AARC_RATE.append(np.mean(AARC[-1000:]))
        #AARC_RATE.append(np.mean(AARC))
        ORC.append(list_ob_col[i])#/(time_steps[i] * num_agents))
        #ORC_RATE.append(np.mean(ORC))
        ORC_RATE.append(np.mean(ORC[-1000:]))
        URC.append(list_un_col[i])#/(time_steps[i] * num_agents))
        URC_RATE.append(np.mean(URC[-1000:]))


        # print('Yo', list_task_completion[i])

        crt_sum += list_task_completion[i]
        #CRT.append(crt_sum / (i + 1))
        if i + 1 >= 1000:
            CRT.append(crt_sum / 1000)
        else:
            CRT.append(crt_sum / (i + 1))
        ctt_sum += time_steps[i]
        #CTT.append(ctt_sum / (i + 1))
        if i + 1 >= 1000:
            CTT.append(ctt_sum / 1000)
        else:
            CTT.append(ctt_sum / (i + 1))

    plt.plot(ind, AVG_SCORE)
    plt.title('Average Score Per 1000 epi')
    plt.xlabel('Number of Episodes')
    plt.ylabel('Average Score')
    plt.show()

    plt.plot(ind, AARC_RATE)
    plt.title('Agent-Agent Collision Rate Per 1000 Epi')
    plt.xlabel('Number of Episodes')
    plt.ylabel('AARC')
    plt.show()

    plt.plot(ind, ORC_RATE)
    plt.title('Agent-Obstacle Collision Rate Per 1000 Epi')
    plt.xlabel('Number of Episodes')
    plt.ylabel('ORC')
    plt.show()

    plt.plot(ind, URC_RATE)
    plt.title('Agent-Rough Terrain Collision Rate Per 1000 Epi')
    plt.xlabel('Number of Episodes')
    plt.ylabel('URC')
    plt.show()

    plt.plot(ind, CRT)
    plt.title('Task-Completion Rate Per 1000 Epi')
    plt.xlabel('Number of Episodes')
    plt.ylabel('CRT')
    plt.show()

    plt.plot(ind, score_history)
    plt.title('Score')
    plt.xlabel('Number of Episodes')
    plt.ylabel('Score')
    plt.show()

    plt.plot(ind, CTT)
    plt.title('Time Steps Taken to Complete Tasks Per 1000 Epi')
    plt.xlabel('Number of Episodes')
    plt.ylabel('CTT')
    plt.show()