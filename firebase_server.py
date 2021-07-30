import os
import numpy as np
from matplotlib import pyplot as plt
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as tt
from torchvision.utils import save_image
from torchvision.transforms import Compose
import numpy as np
import torch.nn as nn
from torch.nn.utils import spectral_norm
import PIL
from PIL import Image
import math
import io
from io import StringIO
import base64
import sys
from base64 import encodebytes
import random
import torchvision.transforms as transforms
from torchvision.utils import save_image
from floorplan_dataset_maps import FloorplanGraphDataset, floorplan_collate_fn
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn.functional as F
import torch.autograd as autograd
from PIL import Image, ImageDraw
from reconstruct import reconstructFloorplan
import svgwrite
from utils import bb_to_img, bb_to_vec, bb_to_seg, mask_to_bb, remove_junctions, ID_COLOR, bb_to_im_fid
from models import Generator
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
import cv2
import webcolors
import sys
import requests
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
# from Model import UNet 
import pickle 
from flask import Flask, flash, render_template, request, url_for
# please note the import from `flask_uploads` - not `flask_reuploaded`!!
# this is done on purpose to stay compatible with `Flask-Uploads`
from flask_uploads import IMAGES, UploadSet, configure_uploads
from flask_cors import CORS
from hgan import one_hot_encoding, process_edges, custom_input, floorplan_collate_fn, pad_im, draw_graph, draw_masks, draw_floorplan
from werkzeug.utils import secure_filename
import gdown
# # Make url public for colab
# from flask_ngrok import run_with_ngrok

if not os.path.isfile("./housegan_pickle.sav"):
    url = 'https://drive.google.com/uc?id=1rRCQPDX0kFPqyJMfc-I-xXa9ly5FlX-L'
    output = 'housegan_pickle.sav'
    gdown.download(url, output, quiet=False) 

app = Flask(__name__, static_folder='generated')
photos = UploadSet("photos", IMAGES)
app.config["UPLOADED_PHOTOS_DEST"] = "images"
app.config["SECRET_KEY"] = os.urandom(24)
configure_uploads(app, photos)


UPLOAD_FOLDER = './images'
ALLOWED_EXTENSIONS = {'png'}

# app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


test_dir ="./"
# For colab Start ngrok when the app is running
# run_with_ngrok(app)

CORS(app)

# FIREBASE
cred = credentials.Certificate('./firebase_cre.json')
# cred = credentials.RefreshToken('./firebase_cre.json')
firebase_admin.initialize_app(cred, {
                'storageBucket': 'housegan-3f845.appspot.com'
            })
bucket = storage.bucket()

# Start ngrok when the app is running
# run_with_ngrok(app)

def denorm(img_tensor):
    return img_tensor


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
# Convert to conversion
conversion = {1: 'living', 2: 'kitchen', 3: 'bedroom', 
              4: 'bathroom', 5:'closet', 6:'balcony', 7: 'corridor', 
              8: 'dining', 9: 'laundry', 10: 'unkown'}
ulta = {conversion[i]:i for i in conversion.keys()}

nodes = ['living',
         'bedroom',
         'bedroom',
         'bathroom',
         'bathroom',
         'balcony',
         'unkown',
         'dining'
]
edges = [
         [0],
         [0, 1],
         [0, 2],
         [0, 6],
         [0, 7],
         [1],
         [1, 0],
         [1, 3],
         [1, 5],
         [1,6],
         [2],
         [2, 0],
         [2, 4],
         [3],
         [3, 1],
         [4],
         [4, 2],
         [5],
         [5, 1],
         [6],
         [6, 0],
         [7],
         [7, 0],
          [7]
]

@app.route("/", methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' :
        data = request.get_json(force=True)
        nodes = data['nodes']
        edges = data['edges']
        print(data)
        # print(request.files)
        if True:#'photo' in request.files:

            print("request has photo!!")
            os.system('rm -rf images')
            os.system('rm -rf generated')
            os.system('rm -rf dump')
            os.system('mkdir images')
            os.system('mkdir generated')
            os.system('mkdir dump')
            # Variables
            num_variations = 8
            latent_dim = 128

            # Input Formating
            n, e = one_hot_encoding(nodes), process_edges(nodes, edges)
            # Load Model weights
            filename = 'housegan_pickle.sav'
            generator = pickle.load(open(filename, 'rb'))
            # Initialize variables
            cuda = True if torch.cuda.is_available() else False
            # cuda = False 
            if cuda:
                generator.cuda()
            Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
            # ------------
            #  Vectorize
            # ------------
            final_images = []
            page_count = 0
            n_rows = 0
            # Customize nodes and eds
            nds, eds = custom_input(n, e)
            given_nds = Variable(nds.type(Tensor))
            given_eds = eds

            for k in range(num_variations):
                # plot images
                z = Variable(Tensor(np.random.normal(0, 1, (given_nds.shape[0], latent_dim))))
                with torch.no_grad():
                    gen_mks = generator(z, given_nds, given_eds)
                    gen_bbs = np.array([np.array(mask_to_bb(mk)) for mk in gen_mks.detach().cpu()])
                    real_nodes = np.where(given_nds.detach().cpu()==1)[-1]
                gen_bbs = gen_bbs[np.newaxis, :, :]/32.0
                junctions = np.array(bb_to_vec(gen_bbs))[0, :, :]
                regions = np.array(bb_to_seg(gen_bbs))[0, :, :, :].transpose((1, 2, 0))
                graph = [real_nodes, None]
                
                if k == 0:
                    graph_arr = draw_graph([real_nodes, eds.detach().cpu().numpy()])
                    final_images.append(graph_arr)
                    
                # reconstruct
                fake_im_seg = draw_masks(gen_mks, real_nodes)
                final_images.append(fake_im_seg)
                fake_im_bb = bb_to_im_fid(gen_bbs, real_nodes, im_size=256).convert('RGBA')
                final_images.append(fake_im_bb)
            
            # Save Individual generated plans, xrays and grap
            n_rows += 1
            final_images_new = []
            img_to_save = []
            plan_idx = 0
            for idx, im in enumerate(final_images):
                print(np.array(im).shape, idx)
            
                final_images_new.append(torch.tensor(np.array(im).transpose((2, 0, 1)))/255.0)

                to_save = torch.tensor(np.array(im).transpose((2, 0, 1)))/255.0
                if idx == 0:
                    save_image(to_save, "./generated/graph.png",
                            nrow=2*num_variations+1, padding=2, range=(0, 1), pad_value=0.5, normalize=False)
                elif idx%2:
                    save_image(to_save, "./generated/xray{}.png".format(plan_idx), 
                            nrow=2*num_variations+1, padding=2, range=(0, 1), pad_value=0.5, normalize=False)
                else:
                    save_image(to_save, "./generated/plan{}.png".format(plan_idx), 
                            nrow=2*num_variations+1, padding=2, range=(0, 1), pad_value=0.5, normalize=False)
                    plan_idx += 1

            # Save all generated plans
            final_images = final_images_new
            final_images = torch.stack(final_images)
            save_image(final_images, "./generated/total{}.png".format(page_count), 
                    nrow=2*num_variations+1, padding=2, range=(0, 1), pad_value=0.5, normalize=False)
            page_count += 1
            n_rows = 0
            final_images = []

            total_path_to_save = "generated/total{}.png".format(str(page_count-1))
            gen_path_to_save = "generated/plan0.png"
            flash("Processed Successfully", "p")
            path_to_save = [gen_path_to_save, total_path_to_save]

            # JSON
            # pil_img = Image.open("./generated/plan0.png", mode='r') # reads the PIL image
            # byte_arr = io.BytesIO()
            # pil_img.save(byte_arr, format='PNG') # convert the PIL image to byte array
            # encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
            

            # Firebase
            # image_data = requests.get("http://127.0.0.1:5000/generated/plan0.png").content
            # image_data = Image.open("./generated/plan0.png", mode='r').convert("RGB") #Image.open(io.BytesIO(base64.b64decode(encoded_img)))
            for i in range(num_variations):
                # pil_img = Image.open("./generated/plan{}.png".format(i), mode='r').convert("RGB")
                # pil_img.save("./generated/plan{}.jpg".format(i))

                # with open("./generated/plan{}.jpg".format(i), "rb") as image2string:
                #     converted_string = base64.b64encode(image2string.read())

                bucket = storage.bucket()
                blob = bucket.blob('floorplan/plan{}.jpg'.format(i))
                # blob.upload_from_string(
                #         converted_string,
                #         content_type='image/jpg'
                #         ,metadata: {contentType: "image/png"}
                #     )
                blob.upload_from_filename("./generated/plan{}.png".format(i))
                blob.make_public()
                print(blob.public_url)

            # return encoded_img
            return render_template('upload.html', img_path=path_to_save)

        else:
            return "'photo' not found in form-data!!"

    # return prediction
    print("Its a GET request!!")
    return render_template('upload.html')

if __name__ == "__main__":
    # app.run(debug=True, use_reloader=True, threaded=True)
    app.run(host="0.0.0.0", port=5000)
    # app.run()