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
