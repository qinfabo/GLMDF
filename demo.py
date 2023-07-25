import os
import torch
import torch.nn as nn
import argparse
import time
from dataset_utils import *
import torch.optim as optim
import DiffLoss
from torchvision import transforms
import read_split_data
import torch.optim as optim
from torchvision import transforms
import s_MyDataSet
import MyDataSet
import main_net
import Decoder
from sklearn.metrics import classification_report

def main():
    bs = 300
    nw = 2
    
    epochs = 100
    device = torch.device("cuda:0"if torch.cuda.is_available() else"cpu")
    print("{} is used".format(device))

    vision_path= os.path.abspath(os.path.join("/content/m_vision_sensor","vision_data_42"))
    sensor_path= os.path.abspath(os.path.join("/content/m_vision_sensor","sensor_data_42"))
    train_s_listpath, train_slab_listpath, val_s_listpath, val_slab_listpath,val_scome_label, train_scome_label = read_split_data(sensor_path)
    train_im_listpath, train_lab_listpath, val_im_listpath, val_lab_listpath,val_vcome_label, train_vcome_label = read_split_data(vision_path)
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    s_traindataset =s_MyDataSet( Len=511,
                                images_path=train_s_listpath, 
                                images_class=train_slab_listpath, 
                                come_class=train_scome_label,
                                  transform=None)
    s_valdataset =s_MyDataSet(   Len=511,
                                 images_path=val_s_listpath, 
                                images_class=val_slab_listpath,
                                come_class=val_scome_label, 
                                  transform=None)
    v_traindataset =MyDataSet(images_path=train_im_listpath, 
                                images_class=train_lab_listpath, 
                                come_class=train_vcome_label,
                                  transform=data_transform["train"])
    v_valdataset =MyDataSet(images_path=val_im_listpath, 
                                images_class=val_lab_listpath, 
                                come_class=val_vcome_label,
                                  transform=data_transform["val"])
    
    
   
    v_train = torch.utils.data.DataLoader(  v_traindataset,
                                             batch_size=bs,
                                               shuffle=True,
                                               pin_memory=False,
                                               num_workers=nw,
                                               collate_fn=v_traindataset.collate_fn



    )
    v_val = torch.utils.data.DataLoader(  v_valdataset,
                                             batch_size=bs,
                                               shuffle=True,
                                               pin_memory=False,
                                               num_workers=nw,
                                               collate_fn=v_valdataset.collate_fn



    )
    s_train = torch.utils.data.DataLoader(  s_traindataset,
                                             batch_size=bs,
                                               shuffle=True,
                                               pin_memory=False,
                                               num_workers=nw,
                                               collate_fn=s_traindataset.collate_fn



    )
    
    s_val = torch.utils.data.DataLoader(  s_valdataset,
                                             batch_size=bs,
                                               shuffle=True,
                                               pin_memory=False,
                                               num_workers=nw,
                                               collate_fn=s_valdataset.collate_fn

    )


    net = main_net()
    net.to(device=device)
    v_model_weight_path = "/content/m_vision_sensor/v_pre_resNet34_2.pth"
    assert os.path.exists(v_model_weight_path), "file {} does not exist.".format(v_model_weight_path)
    weights_dict = torch.load(v_model_weight_path, map_location='cpu')
    missing_keys, unexpected_keys = net.p_v_encoder.load_state_dict(weights_dict, strict=False)
    print(missing_keys)
    s_model_weight_path = "/content/m_vision_sensor/v_pre_sensor.pth"
    assert os.path.exists(s_model_weight_path), "file {} does not exist.".format(s_model_weight_path)
    weights_dict_ = torch.load(s_model_weight_path, map_location='cpu')
    missing_keys_, unexpected_keys_ = net.p_s_encoder.load_state_dict(weights_dict_, strict=False)
    print(missing_keys_)
   
    decoder_ = Decoder()
    decoder_.to(device=device)
    params = [p for p in net.parameters() if p.requires_grad]
    decoder_params =[p for p in decoder_.parameters() if p.requires_grad]
   
    optimizer = optim.Adam(params, lr=0.0001)
    optimizer.add_param_group({'params':decoder_params})
    pre_loss = nn.CrossEntropyLoss()
    come_loss = nn.CrossEntropyLoss()
    
    epochlist = [20,40,60,80,100,120,140,160,180]
    best_acc = 0.0
    acc_list =[]
    loss_list = []
    for epoch in range(epochs):
        #train
        net.train()
        decoder_.train()
        if epoch in epochlist:

          for pam in net.parameters():
             pam.requires_grad = False
          for pam in net.mlpcome.parameters():
             pam.requires_grad = True
          for param_net in decoder_.parameters():
             param_net.requires_grad = True
        else:

          for pam in net.parameters():
             pam.requires_grad = True
          for pam in net.mlpcome.parameters():
             pam.requires_grad = False
          for param_net in decoder_.parameters():
             param_net.requires_grad = True
      
        comeloss = 0
        running_loss = 0
        train_bar = tqdm(v_train)
        for step, (v_data,s_data) in enumerate(zip(train_bar,s_train)):
            image, label,come = v_data
            image.to(device)
            label.to(device)
            come.to(device)
            sensor, label_ ,come_=s_data
            sensor.to(device)
            label_.to(device)
            come_.to(device)
            optimizer.zero_grad()
            t1 = time.time()
            diff_loss, sim_loss, decoder_embedding,mlpcome,_ = net.test(image.to(dtype=torch.float, device=device), sensor.to(dtype=torch.float, device=device))
            t2 = time.time()
            logit = decoder_.forw_(decoder_embedding.to(device=device))
            come_logit = come_loss(mlpcome.to(device),come.to(device))
        
            pre_loss_ = pre_loss(logit.to(device), label.to(device))
            
            loss_ = -1*come_logit+1*diff_loss

            loss  = loss_ + pre_loss_
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            comeloss += come_logit.item()

            train_bar.desc = "train epoch[{}/{}] loss{:.3f}".format(epoch+1, epochs, loss)
        comeloss_all = comeloss / len(v_train)
        print(-comeloss_all)
        loss_list.extend([-comeloss_all])
      
        net.eval()
        decoder_.eval()
        acc = 0.0
        decoder_list = []

        pre_sepoch = []
        lab_sepoch = []
        outputs_pt = []
        label_pt = []
        
        save_path = "/content/m_vision_sensor/net_best.pth"
        save_path_ = "/content/m_vision_sensor/decoder_best.pth"
        with torch.no_grad():
              val_bar = tqdm(v_val)
              for step, (v_val_data,s_val_data) in enumerate(zip(val_bar,s_val)):
                  v_val_images, v_val_labels,v_val_come = v_val_data
                  v_val_images.to(device)
                  v_val_labels.to(device)
                  v_val_come.to(device)
                  s_val_sensor, s_val_label,s_val_come = s_val_data
                  s_val_sensor.to(device)
                  s_val_label.to(device)
                  s_val_come.to(device)
                  _, _, val_decoder_embedding,_,_decoder_embedding = net.test(v_val_images.to(dtype=torch.float, device=device), s_val_sensor.to(dtype=torch.float, device=device))
                  decoder_list.extend(_decoder_embedding)
                  outputs = decoder_.forw_(val_decoder_embedding.to(device=device))
                  predict_y = torch.max(outputs, dim=1)[1]
                  ################################
                  pre_sepoch.extend(predict_y.to('cpu'))
                  lab_sepoch.extend(v_val_labels.to('cpu'))
                  outputs_pt.extend(outputs.to("cpu"))
                  label_pt.extend(v_val_labels.to("cpu"))

                  acc += torch.eq(predict_y, v_val_labels.to(device)).sum().item()

                  val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                              epochs)

        val_accurate = acc / len(v_valdataset)

        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f '%
            (epoch + 1, running_loss / len(v_train), val_accurate))
        acc_list.extend([val_accurate])

        print(classification_report(pre_sepoch, lab_sepoch,digits=5))
        if not os.path.isdir('/content/vision_feature'):
            os.makedirs("/content/vision_feature")
        if val_accurate > best_acc:
            best_acc = val_accurate

            torch.save(net.state_dict(), save_path)
            torch.save(decoder_.state_dict(), save_path_)


        print('best_acc: %.4f'%
                  (best_acc))


    print('ok')



if __name__ == "__main__":
   main()
   
   
   
   





  










