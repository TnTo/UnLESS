#!/usr/bin/env python

import pyNetLogo as pyNL #v0.3
import json
import neural_network as nn

def parse_schedule(schedule, nl):
    #parse the json file with instructions

    for action, params in schedule:
        if action == 'set_seed':
            nn.set_seed(params)

        if action == 'create':
            nn.create_nn(params)

        if action == 'load_nn':
            nn.load_nn(params)

        if action == 'load_nl':
            #Create the inteface with NL
            nl = pyNL.NetLogoLink(gui=True, netlogo_home='/opt/netlogo', netlogo_version='6')
            nl.load_model(params['nl_path'])

        if action == 'simulate':
            for parameter, value in params.items():
                #Set parameters on NL
                if type(value) == type(''):
                    nl.command('set {0} "{1}"'.format(parameter, value))
                else:
                    if parameter == 'seed' and value == 0:
                        nl.command('set {0} {1}'.format(parameter, nn.get_random_seed()))
                    else:
                        nl.command('set {0} {1}'.format(parameter, value))
            nl.command('setup')
            nl.command('loop_go')
            nl.command('file-close')

        if action == 'simulate_no_reset':
            for parameter, value in params.items():
                if type(value) == type(''):
                    nl.command('set {0} "{1}"'.format(parameter, value))
                else:
                    if parameter == 'seed' and value == 0:
                        nl.command('set {0} {1}'.format(parameter, nn.get_random_seed()))
                    else:
                        nl.command('set {0} {1}'.format(parameter, value))
            nl.command('setup_parties')
            nl.command('setup_io')
            nl.command('load_nn')
            nl.command('clear-all-plots')
            nl.command('reset-ticks')
            nl.command('loop_go')
            nl.command('file-close')

        if action == 'train':
            nn.train_nn(params)

        if action == 'loop':
            for i in range(params['n_iter']):
                parse_schedule(params['actions'], nl)

if __name__ == "__main__":
    #Create a pointer for NL interface
    nlLink = None
    #Ask path of json file
    schedule_path = input('Insert the path of the schedule to be loaded: ')
    #load json file
    schedule = json.load(open(schedule_path))
    #Parse and execute
    parse_schedule(schedule, nlLink)
