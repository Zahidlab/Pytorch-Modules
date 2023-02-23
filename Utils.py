def show_batch(dl, nmax=64, torch_tensor = True):
  
  if torch_tensor:
    images= dl
  else:
    images, _ = next(iter(dl))
    
  images = to_device(images, "cpu")
  fig,ax = plt.subplots(figsize = (8,8))
  ax.set_xticks([])
  ax.set_yticks([])
  ax.imshow(make_grid( 
      denormalize(images.detach()[:nmax]), nrow = 8).permute(1,2,0)
      
  )
  
  
  def get_name(t, acc = False, model = "CNN"):
    
    root_path = r"C:\Users\HP\Documents\Zahid\OCT\\"
    
    model_save_path =os.path.join(root_path, "Models")
    log_save_path = os.path.join(root_path, "Logs")
    graph_save_path = os.path.join(root_path, "Graphs")
    results_save_path = os.path.join(root_path, "Results")
    
    project_name = "OCT Classifier"
    
    
    if t=="graph":
        if acc:
            uni_name = os.path.join(graph_save_path, f"{model} {project_name} ACC {get_date_time()}.jpg")
        else:
            uni_name = os.path.join(graph_save_path, f"{model} {project_name} {get_date_time()}.jpg")
            
    elif t=="model":
        uni_name = os.path.join(model_save_path, f"{model} {project_name} {get_date_time()}.pth")
    elif t=='log':
        uni_name = os.path.join(log_save_path, f"{model} {project_name} {get_date_time()}.csv")
    elif t== 'results':
        uni_name = os.path.join(results_save_path, f"{model} {project_name} {get_date_time()}.csv")
    else:
        raise ValueError(f'You entered invalid value')
        
        
    return uni_name
