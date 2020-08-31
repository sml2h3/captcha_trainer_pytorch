import os
import fire
import torch


class Main(object):

    def __init__(self):
        pass

    def init(self, project_name):
        from utils.init_project import InitProject
        InitProject(project_name=project_name).make(True)
        pass

    def data(self, project_name, data_path, scale=0.97, words=False):
        from dataset.make_dataset import MakeDataset
        MakeDataset(project_name=project_name, images_path=data_path, word=words).make(scale=scale)
        pass

    def train(self, project_name):
        from framework import FrameWork
        from config import Config
        from dataset.load_dataset import GetLoader
        config = Config(project_name=project_name)
        base_path = config.base_path
        model_path = os.path.join(base_path, project_name, 'models', '{}.pkl'.format(project_name))
        graph_path = os.path.join(base_path, project_name, 'graphs', '{}.onnx'.format(project_name))
        config = config.load_config()

        TEST_STEP = config['Train']['TEST_STEP']
        TARGET = config['Train']['TARGET']
        TARGET_Accuracy = TARGET['Accuracy']
        TARGET_Cost = TARGET['Cost']
        TARGET_Epoch = TARGET['Epoch']
        framework = FrameWork(config=config)
        framework.to(device=framework.device)
        if os.path.exists(model_path):
            print("find model: {}".format(model_path))
            framework.load_state_dict(torch.load(model_path))
        loaders = GetLoader(project_name=project_name).get_loaders()
        train_loader = loaders['train']
        test_loader = iter(loaders['test'])
        epoch = 0
        step = 0
        loss = None
        print("start training")
        while True:
            for idx, (inputs, labels, labels_length) in enumerate(train_loader):
                loss, lr = framework.train_op(inputs=inputs, labels=labels, label_length=labels_length)
                step += 1
                if step % 100 == 0 and step != 0:
                    print('Epoch:', epoch, 'Step', step, '|train loss:' + str(loss), "lr:", lr)
                if step % TEST_STEP == 0 and step != 0:
                    framework.save_model(model_path, framework)
                    try:
                        t_inputs, t_labels, t_labels_length = next(test_loader)
                    except StopIteration:
                        test_loader = iter(loaders['test'])
                        t_inputs, t_labels, t_labels_length = next(test_loader)
                    pred_labels, labels_list, correct_list, error_list = framework.test_op(t_inputs, t_labels,
                                                                                        t_labels_length)
                    accuracy = len(correct_list) / t_inputs.shape[0]
                    print('Epoch:', epoch, 'Step', step, '|train loss:' + str(loss), 'accuracy:', accuracy, "lr:", lr)
                    if accuracy > TARGET_Accuracy and loss < TARGET_Cost:
                        dummy_input = torch.randn(1, framework.image_channel, framework.resize[0], framework.resize[1], device='cuda')
                        input_names = ["input1"]
                        output_names = ["output"]
                        try:
                            framework.cnn.set_swish(memory_efficient=False)
                            framework.cnn.eval()
                        except Exception:
                            pass
                        torch.onnx.export(framework, dummy_input, graph_path, verbose=True, input_names=input_names,
                                          output_names=output_names)
                        print("Training finished!")
                        exit()

            epoch += 1
            framework.scheduler.step()


if __name__ == '__main__':
    fire.Fire(Main)
