def train_model(model, criterion, optimizer,start_epoch, num_epochs):
    
    train_loss_values = np.load('train_loss.npy').tolist()[:start_epoch]
    valid_loss_values = np.load('valid_loss.npy').tolist()[:start_epoch]
    train_accuracy = np.load('train_acc.npy', allow_pickle=True).tolist()[:start_epoch]
    valid_accuracy = np.load('valid_acc.npy', allow_pickle=True).tolist()[:start_epoch]
    
    if not os.path.exists('saved_models/'):
        os.makedirs(os.path.join('saved_models/'))

    since = time.time()
#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_acc = 0.0
    e_count = 0
    for epoch in range(num_epochs):
        e_count+=1  
        if (e_count==25):
            e_count = 0
            torch.save({
            'epoch': start_epoch + epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
#             'scheduler': scheduler
            }, 'saved_models/' + str(start_epoch+ epoch)+'.pth')
        
        print('Epoch {}/{}'.format(start_epoch+ epoch, start_epoch+num_epochs))
        print('-' * 10)
# Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            class_total = list(0. for i in range(5))
            class_correct = list(0. for i in range(5))

            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
        # Iterate over data.
            for inputs, labels in dataloaders[phase]:
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)
    # zero the parameter gradients
                optimizer.zero_grad()
    # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
#                     if torch.isnan(pred_locs).any():
#                       print("Nan in output prediction:", index)
#                       return None
                    
                    _, predicted = torch.max(outputs, 1)
                    c = (predicted == labels).squeeze()
                    loss = criterion(outputs, labels)
                    for i in range(len(labels)):
                        label = labels[i]
                        class_correct[label] += c[i].item()
                        class_total[label] += 1
    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
    # statistics
                running_loss += loss.item()
                running_corrects += torch.sum(predicted == labels.data)
            
            epoch_loss = running_loss / (dataset_sizes[phase])
        
#             if phase == 'train':
#                 scheduler.step()
                    
            epoch_acc = running_corrects.double() / (dataset_sizes[phase])
            print('{} Loss: {:.4f}'.format(
                        phase, epoch_loss))
            print('Accuracy: {:.4f}'.format(epoch_acc), end = '         ')
            for i in range(5):
                print('%5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]), end = '      ')
            print()
            print()
            
            if phase == 'train':
                train_loss_values.append(epoch_loss)
                np.save('train_loss.npy',train_loss_values)
                train_accuracy.append(epoch_acc)
                np.save('train_acc.npy',train_accuracy)

            if phase == 'valid':
                valid_loss_values.append(epoch_loss)
                np.save('valid_loss.npy',valid_loss_values)
                valid_accuracy.append(epoch_acc)
                np.save('valid_acc.npy',valid_accuracy)
        print()
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
#     print('Best val Acc: {:4f}'.format(best_acc))
# #load best model weights
#     model.load_state_dict(copy.deepcopy(model.state_dict()))
    return model