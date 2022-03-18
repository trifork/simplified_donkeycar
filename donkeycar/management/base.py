
import argparse
import os
import shutil
import stat
import sys

import donkeycar as dk
from donkeycar.utils import *

PACKAGE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
TEMPLATES_PATH = os.path.join(PACKAGE_PATH, 'templates')

HELP_CONFIG = 'location of config file to use. default: ./config.py'


def make_dir(path):
    real_path = os.path.expanduser(path)
    print('making dir ', real_path)
    if not os.path.exists(real_path):
        os.makedirs(real_path)
    return real_path


def load_config(config_path):

    '''
    load a config from the given path
    '''
    conf = os.path.expanduser(config_path)

    if not os.path.exists(conf):
        print("No config file at location: %s. Add --config to specify\
                location or run from dir containing config.py." % conf)
        return None

    try:
        cfg = dk.load_config(conf)
    except:
        print("Exception while loading config from", conf)
        return None

    return cfg


class BaseCommand(object):
    pass


class CreateCar(BaseCommand):
    
    def parse_args(self, args):
        parser = argparse.ArgumentParser(prog='createcar', usage='%(prog)s [options]')
        parser.add_argument('--path', default=None, help='path where to create car folder')
        parser.add_argument('--template', default=None, help='name of car template to use')
        parser.add_argument('--overwrite', action='store_true', help='should replace existing files')
        parsed_args = parser.parse_args(args)
        return parsed_args
        
    def run(self, args):
        args = self.parse_args(args)
        self.create_car(path=args.path, template=args.template, overwrite=args.overwrite)
  
    def create_car(self, path, template='complete', overwrite=False):
        """
        This script sets up the folder structure for donkey to work.
        It must run without donkey installed so that people installing with
        docker can build the folder structure for docker to mount to.
        """

        # these are neeeded incase None is passed as path
        path = path or '~/mycar'
        template = template or 'complete'
        print("Creating car folder: {}".format(path))
        path = make_dir(path)
        
        print("Creating data & model folders.")
        folders = ['models', 'data', 'logs']
        folder_paths = [os.path.join(path, f) for f in folders]   
        for fp in folder_paths:
            make_dir(fp)
            
        # add car application and config files if they don't exist
        app_template_path = os.path.join(TEMPLATES_PATH, template+'.py')
        config_template_path = os.path.join(TEMPLATES_PATH, 'cfg_' + template + '.py')
        myconfig_template_path = os.path.join(TEMPLATES_PATH, 'myconfig.py')
        train_template_path = os.path.join(TEMPLATES_PATH, 'train.py')
        calibrate_template_path = os.path.join(TEMPLATES_PATH, 'calibrate.py')
        car_app_path = os.path.join(path, 'manage.py')
        car_config_path = os.path.join(path, 'config.py')
        mycar_config_path = os.path.join(path, 'myconfig.py')
        train_app_path = os.path.join(path, 'train.py')
        calibrate_app_path = os.path.join(path, 'calibrate.py')
        
        if os.path.exists(car_app_path) and not overwrite:
            print('Car app already exists. Delete it and rerun createcar to replace.')
        else:
            print("Copying car application template: {}".format(template))
            shutil.copyfile(app_template_path, car_app_path)
            os.chmod(car_app_path, stat.S_IRWXU)

        if os.path.exists(car_config_path) and not overwrite:
            print('Car config already exists. Delete it and rerun createcar to replace.')
        else:
            print("Copying car config defaults. Adjust these before starting your car.")
            shutil.copyfile(config_template_path, car_config_path)

        if os.path.exists(train_app_path) and not overwrite:
            print('Train already exists. Delete it and rerun createcar to replace.')
        else:
            print("Copying train script. Adjust these before starting your car.")
            shutil.copyfile(train_template_path, train_app_path)
            os.chmod(train_app_path, stat.S_IRWXU)

        if os.path.exists(calibrate_app_path) and not overwrite:
            print('Calibrate already exists. Delete it and rerun createcar to replace.')
        else:
            print("Copying calibrate script. Adjust these before starting your car.")
            shutil.copyfile(calibrate_template_path, calibrate_app_path)
            os.chmod(calibrate_app_path, stat.S_IRWXU)

        if not os.path.exists(mycar_config_path):
            print("Copying my car config overrides")
            shutil.copyfile(myconfig_template_path, mycar_config_path)
            # now copy file contents from config to myconfig, with all lines
            # commented out.
            cfg = open(car_config_path, "rt")
            mcfg = open(mycar_config_path, "at")
            copy = False
            for line in cfg:
                if "import os" in line:
                    copy = True
                if copy: 
                    mcfg.write("# " + line)
            cfg.close()
            mcfg.close()
 
        print("Donkey setup complete.")


class Train(BaseCommand):

    def parse_args(self, args):
        HELP_FRAMEWORK = 'the AI framework to use (tensorflow|pytorch). ' \
                         'Defaults to config.DEFAULT_AI_FRAMEWORK'
        parser = argparse.ArgumentParser(prog='train', usage='%(prog)s [options]')
        parser.add_argument('--tub', nargs='+', help='tub data for training')
        parser.add_argument('--model', default=None, help='output model name')
        parser.add_argument('--type', default=None, help='model type')
        parser.add_argument('--config', default='./config.py', help=HELP_CONFIG)
        parser.add_argument('--framework',
                            choices=['tensorflow', 'pytorch', None],
                            required=False,
                            help=HELP_FRAMEWORK)
        parser.add_argument('--checkpoint', type=str,
                            help='location of checkpoint to resume training from')
        parser.add_argument('--transfer', type=str, help='transfer model')
        parser.add_argument('--comment', type=str,
                            help='comment added to model database - use '
                                 'double quotes for multiple words')
        parsed_args = parser.parse_args(args)
        return parsed_args

    def run(self, args):
        args = self.parse_args(args)
        args.tub = ','.join(args.tub)
        cfg = load_config(args.config)
        framework = args.framework if args.framework \
            else getattr(cfg, 'DEFAULT_AI_FRAMEWORK', 'tensorflow')

        if framework == 'tensorflow':
            from donkeycar.pipeline.training import train
            train(cfg, args.tub, args.model, args.type, args.transfer,
                  args.comment)
        elif framework == 'pytorch':
            from donkeycar.parts.pytorch.torch_train import train
            train(cfg, args.tub, args.model, args.type,
                  checkpoint_path=args.checkpoint)
        else:
            print(f"Unrecognized framework: {framework}. Please specify one of "
                  f"'tensorflow' or 'pytorch'")


def execute_from_command_line():
    """
    This is the function linked to the "donkey" terminal command.
    """
    commands = {
        'createcar': CreateCar,
        'train': Train,
    }
    
    args = sys.argv[:]

    if len(args) > 1 and args[1] in commands.keys():
        command = commands[args[1]]
        c = command()
        c.run(args[2:])
    else:
        dk.utils.eprint('Usage: The available commands are:')
        dk.utils.eprint(list(commands.keys()))

    
if __name__ == "__main__":
    execute_from_command_line()
