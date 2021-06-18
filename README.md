

Download the pickle formated model and place it on the same directory.
https://drive.google.com/uc?id=1Xox9t_1RfWlFSIWxIJD4U_vUx55mfy1r
OR

Download model weights and uncomment torch.load and Unet in the script
https://drive.google.com/file/d/1c-c0f4zjQfZwykzRpTnc2rA0jnkfckdU/view?usp=sharing

<h1> Files for deployment GCP </h1>

1. main.py
2. pickle formated model (main.py can download it, if not present)
3. requirements.txt
4. app.yaml
5. templates

<h1> Files Description </h1>

1. Run app.py, for running the model on local system.
2. main.py is for GCP server (writen in flask)
3. cloab_server.py is server file for running in colab  (install flask_ngrok for running flask on colab)

<h2> Colab_nootbook for running demo on colab </h2>
https://github.com/tejasbana/floorPlan/blob/main/colab_nootbook.ipynb
