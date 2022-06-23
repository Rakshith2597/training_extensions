'''
This file contains the functions for training, validating and testing the models.
'''

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import random
from parameter import zSize, batchSize, iteration, learningRate, alpha, gamma, gamma2, beta, delta, numClasses
from utils.lenet import Encoder, Classifier1, Discriminator  # file
from imageLoader1 import Dataloder_img  # file
from parameter import trainingDataPath, testDataPath, dataStorePath
from vectorHandle import shuffler  # file


class Trainer():
    '''
    XYZ
    '''

    def __init__(self, trainloader, testloader, start, encoder, classifier1, discriminator, discriminator_criterion, discriminator_optimizer, classifier_optimizer, ac1_loss_criterion, loss_criterion, encoder_optimizer, similarity_loss_dict, test_similarity_loss_dict, relational_loss_dict, test_relational_loss_dict, classifier_loss_dict, discriminator_loss_dict, hd_item, discTestAccuracyDict, classifierTestAccuracyDict, classes):
        '''
        XYZ
        '''
        self.trainloader = trainloader
        self.testloader = testloader
        self.start = start
        self.encoder = encoder
        self.classifier1 = classifier1
        self.discriminator = discriminator
        self.discriminator_criterion = discriminator_criterion
        self.discriminator_optimizer = discriminator_optimizer
        self.classifier_optimizer = classifier_optimizer
        self.ac1_loss_criterion = ac1_loss_criterion
        self.loss_criterion = loss_criterion
        self.encoder_optimizer = encoder_optimizer
        self.classes = classes

        self.similarity_loss_dict = similarity_loss_dict
        self.test_similarity_loss_dict = test_similarity_loss_dict
        self.relational_loss_dict = relational_loss_dict
        self.test_relational_loss_dict = test_relational_loss_dict
        self.classifier_loss_dict = classifier_loss_dict
        self.discriminator_loss_dict = discriminator_loss_dict
        self.hd_item = hd_item
        self.discTestAccuracyDict = discTestAccuracyDict
        self.classifierTestAccuracyDict = classifierTestAccuracyDict

        self.curr_epoch = 0

    def train(self):
        for epoch in range(self.start, iteration):
            similarity_loss = 0
            similarity_loss_temp = 0
            similarity_count_temp = 0
            similarity_count_full = 0

            relational_loss = 0
            relational_loss_temp = 0
            relational_count_temp = 0
            relational_count_full = 0

            classifier_loss = 0
            classifier_loss_temp = 0
            classifier_count_temp = 0
            classifier_count_full = 0

            discriminator_loss = 0
            discriminator_loss_temp = 0
            discriminator_count_temp = 0
            discriminator_count_full = 0

            hd_t0 = 0
            hd_t1 = 0
            hd_t2 = 0

            self.curr_epoch = epoch

            for i, data in enumerate(self.trainloader, 0):
                similarity_loss, similarity_loss_temp, similarity_count_temp, similarity_count_full, relational_loss, relational_loss_temp, relational_count_temp, relational_count_full, classifier_loss, classifier_loss_temp, classifier_count_temp, classifier_count_full, discriminator_loss, discriminator_loss_temp, discriminator_count_temp, discriminator_count_full, hd_t0, hd_t1, hd_t2 = self.epoch_train(
                    data, i, similarity_loss, similarity_loss_temp, similarity_count_temp, similarity_count_full, relational_loss, relational_loss_temp, relational_count_temp, relational_count_full, classifier_loss, classifier_loss_temp, classifier_count_temp, classifier_count_full, discriminator_loss, discriminator_loss_temp, discriminator_count_temp, discriminator_count_full, hd_t0, hd_t1, hd_t2)

                test_similarity_loss_mean, test_relational_loss_mean, classifierTestAccuracyMean, discTestAccuracyMean = self.test()
                print("accuraccy for epoch", epoch, ":", test_similarity_loss_mean,
                      test_relational_loss_mean, classifierTestAccuracyMean, discTestAccuracyMean)

    def epoch_train(self, data, i, similarity_loss, similarity_loss_temp, similarity_count_temp, similarity_count_full, relational_loss, relational_loss_temp, relational_count_temp, relational_count_full, classifier_loss, classifier_loss_temp, classifier_count_temp, classifier_count_full, discriminator_loss, discriminator_loss_temp, discriminator_count_temp, discriminator_count_full, hd_t0, hd_t1, hd_t2):
        # if i > 4 :
        # break
        input1, input2, labels, groundtruths1, groundtruths2 = data

        indexes0 = np.where(labels.numpy() == 0)[0].tolist()
        indexes1 = np.where(labels.numpy() == 1)[0].tolist()
        indexes2 = np.where(labels.numpy() == 2)[0].tolist()
        # print(len(indexes0))
        if not len(indexes2) == batchSize:
            input1_new = torch.from_numpy(
                np.delete(input1.numpy(), indexes2, 0))
            input2_new = torch.from_numpy(
                np.delete(input2.numpy(), indexes2, 0))
            labels_2 = 1-labels[labels != 2]
            input1_new, input2_new, labels_2 = Variable(input1_new).cuda(
            ), Variable(input2_new).cuda(), Variable(labels_2).cuda()
            h1_new, _ = self.encoder(input1_new)
            h2_new, _ = self.encoder(input2_new)

        input1, input2, labels = Variable(input1).cuda(), Variable(
            input2).cuda(), Variable(labels).cuda()
        groundtruths1, groundtruths2 = Variable(
            groundtruths1).cuda(), Variable(groundtruths2).cuda()
        # print(input1.shape)
        h1, x1 = self.encoder(input1)
        h2, x2 = self.encoder(input2)
        # print(h1.shape)
        # print(h2.shape)
        #########################
        ##### Discriminator #####
        #########################
        if len(indexes0) > 0:
            d_h1 = h1[indexes0]
            d_h2 = h2[indexes0]
            d_h1, d_h2, dlabels = shuffler(d_h1, d_h2)
            dlabels = Variable(dlabels).cuda()
            d_input = torch.stack((d_h1, d_h2), 1)
            # print("1",d_input.shape)
            d_output = self.discriminator(d_input).view(-1)
            # print(d_output.shape)
            d_loss = alpha * \
                self.discriminator_criterion(dlabels.float(), d_output)

            self.discriminator_optimizer.zero_grad()
            d_loss.backward(retain_graph=True)
            self.discriminator_optimizer.step()

            discriminator_count_temp += 1
            discriminator_count_full += 1

        #################################
        ##### AUXILARY CLASSIFIER1  #####
        #################################

        pred = self.classifier1(x1)
        ac1_loss = self.ac1_loss_criterion(pred, groundtruths1)

        self.classifier1_optimizer.zero_grad()
        # encoder_optimizer.zero_grad()

        ac1_loss.backward(retain_graph=True)

        self.classifier1_optimizer.step()
        # encoder_optimizer.step()

        classifier_count_temp += 1
        classifier_count_full += 1

        ###########################################
        ##### CAUCHY LOSS 1 [t =2 vs t=1,t=0] #####
        ###########################################
        torch.autograd.set_detect_anomaly(True)
        del(input1)
        del(input2)
        s = (labels != 2)
        cos = F.cosine_similarity(h1, h2, dim=1, eps=1e-6)
        dist = F.relu((1-cos)*zSize/2)

        hd_t0 += torch.sum(dist[indexes0]).item() / \
            (dist[indexes0].size(0) + 0.0000001)
        hd_t1 += torch.sum(dist[indexes1]).item() / \
            (dist[indexes1].size(0) + 0.0000001)
        hd_t2 += torch.sum(dist[indexes2]).item() / \
            (dist[indexes2].size(0) + 0.0000001)

        cauchy_output = torch.reciprocal(dist+gamma)*gamma
        try:
            loss1 = delta * \
                self.loss_criterion(torch.squeeze(cauchy_output), s.float())
        except RuntimeError:
            print(torch.squeeze(cauchy_output))
            print(s)
            print("s", torch.max(s.float()).item(),
                  torch.min(s.float()).item())
            print("\nCO ", torch.max(torch.squeeze(cauchy_output)).item(),
                  torch.min(torch.squeeze(cauchy_output)).item())
       # encoder_optimizer.zero_grad()
        # loss.backward()
        # encoder_optimizer.step()

        #######################################
        #####  CAUCHY LOSS 2 [t=1 vs t=0] #####
        #######################################

        if not len(indexes2) == batchSize:
            cos = F.cosine_similarity(h1_new, h2_new, dim=1, eps=1e-6)
            dist = F.relu((1-cos)*zSize/2)
            cauchy_output = torch.reciprocal(dist+gamma)*gamma
            try:
                loss2 = beta * \
                    self.loss_criterion(torch.squeeze(
                        cauchy_output), labels_2.float())
            except RuntimeError:
                print(torch.squeeze(cauchy_output))
                print(labels_2)
                print("s", torch.max(labels_2.float()).item(),
                      torch.min(labels_2.float()).item())
                print("\nCO ", torch.max(torch.squeeze(cauchy_output)).item(),
                      torch.min(torch.squeeze(cauchy_output)).item())

            # encoder_optimizer.zero_grad()
            # loss.backward()
            # encoder_optimizer.step()

            relational_count_temp += 1
            relational_count_full += 1

        # print("\n\n\n gradient started...\n\n")
        # for name,param in encoder.named_parameters():
        #     print(name))
        #     print("Grad Output",torch.max(param.grad),torch.min(param.grad))
        # print("\n\n\n gradient ended...\n\n")

        # loss_count = 250  # number of ITERATIONS after which loss is shown
        # if (i + 1) % loss_count == 0:
        #    print('[%d, %d] Cauchy loss 1: %.3f Cauchy loss 2: %.3f' % (epoch + 1, i + 1, similarity_loss_temp/loss_count, relational_loss_temp/relational_count_temp))
        #    similarity_loss_temp = 0
        #    similarity_count_temp = 0
        #    relational_loss_temp = 0
        #    relational_count_temp = 0
            # run_loss_temp = 0.0

        # key = str((epoch+1))+"_"+str(i)

        # loss_dict[epoch] = run_loss/i
        # run_loss = 0.0
        # similarity_loss = 0.0
        # relational_loss = 0.0

        loss = loss1 + loss2
        self.encoder_optimizer.zero_grad()
        loss.backward()
        self.encoder_optimizer.step()
        # scheduler.step()

        discriminator_loss += d_loss.item()/alpha
        discriminator_loss_temp += d_loss.item()/alpha

        classifier_loss += ac1_loss.item()
        classifier_loss_temp += ac1_loss.item()

        similarity_loss += loss.item()/delta
        similarity_loss_temp += loss.item()/delta

        relational_loss += loss.item()/beta
        relational_loss_temp += loss.item()/beta

        self.similarity_loss_dict[self.curr_epoch] = similarity_loss/i
        self.relational_loss_dict[self.curr_epoch] = relational_loss / \
            relational_count_full
        self.lassifier_loss_dict[self.curr_epoch] = classifier_loss / \
            classifier_count_full
        self.discriminator_loss_dict[self.curr_epoch] = discriminator_loss / \
            discriminator_count_full

        ###############################

        #SAVING WEIGHTS AND LOSS FILES#

        ###############################

        encoder_path = os.path.join(
            dataStorePath, 'encoder-%d.pkl' % (self.epoch+1))
        torch.save(self.encoder.state_dict(), encoder_path)
        # print("Saving encoder weights to ", encoder_path)

        classifier_path = os.path.join(
            (dataStorePath), 'classifier-%d.pkl' % (self.epoch+1))
        torch.save(self.classifier1.state_dict(), classifier_path)
        # print("Saving classifier weights to ", classifier_path)

        disc_path = os.path.join(dataStorePath, 'disc-%d.pkl' % (self.epoch+1))
        torch.save(self.discriminator.state_dict(), disc_path)
        # print("Saving discriminator weights to ", disc_path)

        self.hd_item[self.epoch] = (hd_t0/i, hd_t1/i, hd_t2/i)
        hd_path = os.path.join(dataStorePath, 'hd_log.pkl')
        with open(hd_path, 'wb') as handle:
            pickle.dump(self.hd_item, handle)
        # print("Saving hamming distance log to ", hd_path)

        loss_log_path = os.path.join(dataStorePath, 'c1_loss_log.pkl')
        with open(loss_log_path, 'wb') as handle:
            pickle.dump(self.similarity_loss_dict, handle)
        # print("Saving Cauchy1 loss log to ", loss_log_path)

        loss_log_path = os.path.join(dataStorePath, 'ac1_loss_log.pkl')
        with open(loss_log_path, 'wb') as handle:
            pickle.dump(self.classifier_loss_dict, handle)
        # print("Saving Auxilary Classifier loss log to ", loss_log_path)

        loss_log_path = os.path.join(dataStorePath, 'c2_loss_log.pkl')
        with open(loss_log_path, 'wb') as handle:
            pickle.dump(self.relational_loss_dict, handle)
        # print("Saving Cauchy2 loss log to ", loss_log_path)

        loss_log_path = os.path.join(dataStorePath, 'disc_loss_log.pkl')
        with open(loss_log_path, 'wb') as handle:
            pickle.dump(self.discriminator_loss_dict, handle)
        # print("Saving Discriminator loss log to ", loss_log_path)

        return similarity_loss, similarity_loss_temp, similarity_count_temp, similarity_count_full, relational_loss, relational_loss_temp, relational_count_temp, relational_count_full, classifier_loss, classifier_loss_temp, classifier_count_temp, classifier_count_full, discriminator_loss, discriminator_loss_temp, discriminator_count_temp, discriminator_count_full, hd_t0, hd_t1, hd_t2

    def test(self):
        if (self.curr_epoch + 1) % 1 == 0:
            correct_pred = {classname: 0 for classname in self.classes}
            total_pred = {classname: 0 for classname in self.classes}
            confusion_matrix = torch.zeros(8, 8)
            with torch.no_grad():
                t_loss1 = 0
                t_loss2 = 0
                t_run = 0
                ac1_total = 0
                ac1_correct = 0
                d_total = 0
                d_correct = 0

                print('\n Testing ....')
                for t_i, t_data in enumerate(self.testloader):
                    # if t_i > 10:
                    #     break

                    t_input1, t_input2, t_labels, t_gt1, t_gt2 = t_data
                    t_indexes = np.where(t_labels.numpy() == 0)[0].tolist()
                    t_indexes2 = np.where(t_labels.numpy() == 2)[0].tolist()

                    if not len(t_indexes2) == int(batchSize/2):
                        t_input1_new = torch.from_numpy(
                            np.delete(t_input1.numpy(), t_indexes2, 0))
                        t_input2_new = torch.from_numpy(
                            np.delete(t_input2.numpy(), t_indexes2, 0))
                        t_labels_2 = 1-t_labels[t_labels != 2]
                        t_input1_new, t_input2_new, t_labels_2 = Variable(t_input1_new).cuda(
                        ), Variable(t_input2_new).cuda(), Variable(t_labels_2).cuda()
                        h1_t_new, _ = self.encoder(t_input1_new)
                        h2_t_new, _ = self.encoder(t_input2_new)

                    t_input1, t_input2, t_labels = Variable(t_input1).cuda(
                    ), Variable(t_input2).cuda(), Variable(t_labels).cuda()
                    t_gt1, t_gt2 = Variable(
                        t_gt1).cuda(), Variable(t_gt2).cuda()

                    s_t = (t_labels != 2)

                    h1_t, x1_t = self.encoder(t_input1)
                    h2_t, x2_t = self.encoder(t_input2)
                    # disc accuracy
                    if len(t_indexes) > 0:
                        d_h1_t = h1_t[t_indexes]
                        d_h2_t = h2_t[t_indexes]
                        d_h1_t, d_h2_t, dlabels_t = shuffler(d_h1_t, d_h2_t)
                        dlabels_t = Variable(dlabels_t).cuda()
                        d_input_t = torch.stack((d_h1_t, d_h2_t), 1)
                        d_output_t = self.discriminator(d_input_t).view(-1)
                        new_d_output_t = d_output_t > 0.5
                        d_total += len(new_d_output_t)
                        d_correct += len(new_d_output_t) - (new_d_output_t ^
                                                            dlabels_t.byte()).sum().cpu().numpy()

                    # auxilary_classifier1 accuracy
                    t_pred = self.classifier1(x1_t)
                    _, predicted = torch.max(t_pred.data, 1)
                    ac1_total += len(t_gt1)
                    ac1_correct += (predicted == t_gt1).sum().cpu().numpy()
                    '''
                    acc_all = (predicted == t_gt1).float().mean()
                    
                    acc = [0 for c in range(len(classes))]
                    for c in range(len(classes)):
                        #print(c)
                        acc[c] = ((predicted == t_gt1) * (t_gt1 == c)).float() / torch.Tensor(max(t_gt1 == c).sum(), 1)
                    '''
                    '''for t_gt1, prediction in zip(t_gt1, predicted):
                    if t_gt1 == prediction:
                        correct_pred[classes[t_gt1]] += 1
                    total_pred[classes[t_gt1]] += 1

                    for t, p in zip(t_gt1.view(-1), predicted.view(-1)):
                            confusion_matrix[t.long(), p.long()] += 1'''

                    cos = F.cosine_similarity(h1_t, h2_t, dim=1, eps=1e-6)
                    dist = F.relu((1-cos)*zSize/2)
                    cauchy_output = torch.reciprocal(dist+gamma)*gamma

                    try:
                        t_loss1 += self.loss_criterion(
                            (cauchy_output), s_t.float()).item()
                    except RuntimeError:
                        print(torch.squeeze(cauchy_output))
                        print(s_t)
                        print("s", torch.max(s_t.float()).item(),
                              torch.min(s_t.float()).item())
                        print("\nCO ", torch.max(torch.squeeze(cauchy_output)).item(
                        ), torch.min(torch.squeeze(cauchy_output)).item())

                    if not len(t_indexes2) == int(batchSize/2):
                        cos = F.cosine_similarity(
                            h1_t_new, h2_t_new, dim=1, eps=1e-6)
                        dist = F.relu((1-cos)*zSize/2)
                        cauchy_output = torch.reciprocal(dist+gamma)*gamma

                        try:
                            t_loss2 += self.loss_criterion(
                                (cauchy_output), t_labels_2.float()).item()
                        except RuntimeError:
                            print(torch.squeeze(cauchy_output))
                            print(s_t)
                            print("s", torch.max(t_labels_2.float()).item(),
                                  torch.min(t_labels_2.float()).item())
                            print("\nCO ", torch.max(torch.squeeze(cauchy_output)).item(
                            ), torch.min(torch.squeeze(cauchy_output)).item())
                        t_run += 1

                    del(t_input1)
                    del(t_input2)

                #print('Classification Accuracy of the network on the test images: %.5f \n\n' % ((100.0*correct) / total))
                print('Classification Accuracy over test images: %.5f' %
                      ((100.0*ac1_correct) / ac1_total))
                print('Discriminator Accuracy over test images: %.5f' %
                      ((100.0*d_correct) / d_total))
                # print('C2 Classification Accuracy of the network on the test images: %.5f' % ((100.0*c2_correct) / c2_total))

                self.test_similarity_loss_dict[self.curr_epoch] = t_loss1/t_i
                print("Testing loss1: %.5f" % (t_loss1/t_i))
                test_loss_log_path = os.path.join(
                    dataStorePath, 'test_loss_c1_log.pkl')
                with open(test_loss_log_path, 'wb') as handle:
                    pickle.dump(self.test_similarity_loss_dict, handle)

                self.test_relational_loss_dict[self.curr_epoch] = t_loss2/t_run
                print("Testing loss2: %.5f\n" % (t_loss2/t_run))
                test_loss_log_path = os.path.join(
                    dataStorePath, 'test_loss_c2_log.pkl')
                with open(test_loss_log_path, 'wb') as handle:
                    pickle.dump(self.test_relational_loss_dict, handle)

                self.classifierTestAccuracyDict[self.curr_epoch] = (
                    100.0*ac1_correct) / ac1_total
                ac1_accuracy_log_path = os.path.join(
                    dataStorePath, 'ac1_acc_log.pkl')
                with open(ac1_accuracy_log_path, 'wb') as handle:
                    pickle.dump(self.classifierTestAccuracyDict, handle)

                self.discTestAccuracyDict[self.curr_epoch] = (
                    100.0*d_correct) / d_total
                d_accuracy_log_path = os.path.join(
                    dataStorePath, 'disc_acc_log.pkl')
                with open(d_accuracy_log_path, 'wb') as handle:
                    pickle.dump(self.discTestAccuracyDict, handle)

            test_similarity_loss_mean = np.array(
                self.test_similarity_loss_dict).mean()
            test_relational_loss_mean = np.array(
                self.test_relational_loss_dict).mean()
            classifierTestAccuracyMean = np.array(
                self.classifierTestAccuracyDict).mean()
            discTestAccuracyMean = np.array(self.discTestAccuracyDict).mean()

            return test_similarity_loss_mean, test_relational_loss_mean, classifierTestAccuracyMean, discTestAccuracyMean

    def main():
        np.seterr(divide='ignore',
                  invalid='ignore'
                  )
        torch.backends.cudnn.benchmark = True

        #from ray import tune
        #from ray.tune import CLIReporter
        #from ray.tune.schedulers import ASHAScheduler

        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

        encoder = Encoder(zSize)

        classifier1 = Classifier1(numClasses)
        discriminator = Discriminator(zSize)
        if torch.cuda.is_available():
            encoder.cuda()
            classifier1.cuda()
            discriminator.cuda()
        # print(encoder)

        load_model = False

        if load_model:
            print("\n\n Loading Pretrained Model ....... \n\n\n")
            start = 1
            model_path = "/storage/asim/trainingDmcCD/extra/encoder-100.pkl"
            encoder.load_state_dict(torch.load(model_path))
            '''Classifier1.load_state_dict(torch.load("/storage/asim/trainingDmcCD/train(0.00005)/classifier1-1.pkl"))
            Classifier1.load_state_dict(torch.load("/storage/asim/trainingDmcCD/train(0.00005)/classifier2-1.pkl"))
            loss_dict = pickle.load(open("/storage/asim/trainingDmcCD/train(0.00005)/loss_log.pkl", "rb"))
            similarity_loss_dict = pickle.load(open("/storage/asim/trainingDmcCD/train(0.00005)/c1_loss_log.pkl", "rb"))
            relational_loss_dict = pickle.load(open("/storage/asim/trainingDmcCD/train(0.00005)/c2_loss_log.pkl", "rb"))
            c1_acc_dict = pickle.load(open("/storage/asim/trainingDmcCD/train(0.00005)/c1_acc_log.pkl", "rb"))
            c2_acc_dict = pickle.load(open("/storage/asim/trainingDmcCD/train(0.00005)/c2_acc_log.pkl", "rb"))
            test_loss_dict = pickle.load(open("/storage/asim/trainingDmcCD/train(0.00005)/test_loss_log.pkl", "rb"))
            hd_item = pickle.load(open("/storage/asim/trainingDmcCD/train(0.00005)/hd_log.pkl", "rb"))
        '''
        else:
            model_path = '/storage/asim/trainingDmcCD/extra/encoder-100.pkl'
            encoder.load_state_dict(torch.load(model_path), strict=False)
            start = 0
            similarity_loss_dict = {}
            test_similarity_loss_dict = {}
            relational_loss_dict = {}
            test_relational_loss_dict = {}
            classifier_loss_dict = {}
            discriminator_loss_dict = {}
            hd_item = {}
            discTestAccuracyDict = {}
            classifierTestAccuracyDict = {}

        if not os.path.exists(dataStorePath):
            os.makedirs(dataStorePath)

        print("Log to be stored in :", dataStorePath)

        plt.ion()

        # torch.use_deterministic_algorithms(True)
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(0)

        transform = transforms.Compose([
            transforms.Resize((28, 28), interpolation=2),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        trainset = Dataloder_img(
            trainingDataPath, transform=transform, target_transform=None)
        print(len(trainset))
        #sampler = oversample(trainingDataPath)
        # print(len(sampler))

        trainloader = torch.utils.data.DataLoader(
            trainset, shuffle=True,  batch_size=batchSize, num_workers=4)
        testset = Dataloder_img(
            testDataPath, transform=transform, target_transform=None)
        testloader = torch.utils.data.DataLoader(
            testset, shuffle=True, batch_size=int(batchSize/2), num_workers=4)
        print(len(testset))
        print("\nDataset generated. \n\n")

        loss_criterion = nn.BCELoss()
        ac1_loss_criterion = nn.NLLLoss()
        discriminator_criterion = nn.BCEWithLogitsLoss()

        encoder_optimizer = optim.Adam(
            encoder.parameters(), lr=learningRate, eps=0.0001, amsgrad=True)
        classifier1_optimizer = optim.Adam(
            classifier1.parameters(), lr=learningRate, eps=0.0001, amsgrad=True)
        discriminator_optimizer = optim.Adam(
            discriminator.parameters(), lr=learningRate/10, eps=0.0001, amsgrad=True)
        #scheduler = optim.lr_scheduler.StepLR(encoder_optimizer, step_size=10, gamma=0.1)

        encoder.train()

        classes = []
        for d in os.listdir(trainingDataPath):
            if(d != 'New_class'):
                classes.append(d)
        classes = sorted(classes)
        # classes.append('New_class')
        cls_to_idx = {classes[i]: i for i in range(len(classes))}
        # print(cls_to_idx)
        start = 1

        trainer = Trainer(trainloader, testloader, start, encoder, classifier1, discriminator, discriminator_criterion, discriminator_optimizer, classifier1_optimizer, ac1_loss_criterion, loss_criterion, encoder_optimizer,
                          similarity_loss_dict, test_similarity_loss_dict, relational_loss_dict, test_relational_loss_dict, classifier_loss_dict, discriminator_loss_dict, hd_item, discTestAccuracyDict, classifierTestAccuracyDict, classes)
