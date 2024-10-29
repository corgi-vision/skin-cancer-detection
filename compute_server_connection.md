# GCP compute server connection

> [!CAUTION]
>
> ## Always stop the VM after a computing (~1-4 $/hour)



### Add emails

1. Need to add email addresses to the project first

### GClound init

1. **Install GCoud CLI**: https://cloud.google.com/sdk/docs/install

2. ```bash
   gcloud init
   # 2: Create new configuration
   # Name it (e.g.: corgi-vision)
   # Sign in with a new Google Account (choose the registered one)
   # Select the project (deeplearning-436520)
   # Don't need to set up a zone
   
   gcloud compute ssh
   ```

3. **Start the VM **: `gcloud compute instances start jupyter-server`

4. ```bash
   gcloud compute ssh jupyter-server -NL 6001:0.0.0.0:6006
   # if error, start the VM first
   # after that you access to the terminal. 
   ```

5. **Install/Open VSCode**

   1. Install the **Remote - SSH package**, and restart
      <img src="/home/scsng/.config/Typora/typora-user-images/image-20241016112909894.png" alt="image-20241016112909894" style="zoom:40%;" />
   2. Click on the SSH connection button
      <img src="/home/scsng/.config/Typora/typora-user-images/image-20241016113201003.png" alt="image-20241016113201003" style="zoom:10%;" />

 3. Connect to host -> jupyter-server.europe...
    <img src="/home/scsng/.config/Typora/typora-user-images/image-20241016113427890.png" alt="image-20241016113427890" style="zoom:45%;" />

 4. Open file -> show local -> select file

 5. Select Kernel (Top right)
    <img src="/home/scsng/.config/Typora/typora-user-images/image-20241016113753613.png" alt="image-20241016113753613" style="zoom:33%;" />

 6. *(This only needs to do one in a VSCode env.)*
      http://127.0.0.1:8888/tree?token={custom token}
    name it (e. g.:corgi-vision), later it's enough to just click on that

 7. Done, start the development

    > [!TIP]
    >
    > You can access to the TensorBoard here: http://http://localhost:6001/

    > [!CAUTION]
    >
    > ## Always stop the VM after a computing (~1-4 $/hour)
    >
    > The free credits are limited
    >
    > `sudo shutdown 0` in the instance
    >
    > gcloud compute instances stop jupyter-server

    