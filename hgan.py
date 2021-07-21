import argparse
import os
import numpy as np
import math
import sys
import random
import pickle

import torchvision.transforms as transforms
from torchvision.utils import save_image

from floorplan_dataset_maps import FloorplanGraphDataset, floorplan_collate_fn
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
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

def one_hot_encoding(nodes):
    length = 10
    ans = []
    for k in nodes:
        temp = [0 if i != ulta[k] else 1 for i in range(1, length+1)]
        ans.append(temp)
    return torch.Tensor(ans)
def process_edges(nodes, edges):
    prep_edgs = []
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if j > i:
                if [i, j] in edges or [j, i] in edges:
                    prep_edgs.append([i, 1, j])
                else:
                    prep_edgs.append([i, -1, j])
    return torch.tensor(prep_edgs, dtype=torch.int32)


conversion = {1: 'living', 2: 'kitchen', 3: 'bedroom', 
              4: 'bathroom', 5:'closet', 6:'balcony', 7: 'corridor', 
              8: 'dining', 9: 'laundry', 10: 'unkown'}
ulta = {conversion[i]:i for i in conversion.keys()}


def custom_input(n, e):
    return n, e


def floorplan_collate_fn(nds,eds):
	all_rooms_mks, all_nodes, all_edges = [], [], []
	all_node_to_sample, all_edge_to_sample = [], []
	node_offset = 0
	for i, (nodes, edges) in enumerate(zip(nds,eds)):
		O, T = nodes.size(0), edges.size(0)
		# all_rooms_mks.append(rooms_mks)
		all_nodes.append(nodes)
		edges = edges.clone()
		if edges.shape[0] > 0:
			# edges[:, 0] += node_offset
			# edges[:, 2] += node_offset
			all_edges.append(edges)
		all_node_to_sample.append(torch.LongTensor(O).fill_(i))
		all_edge_to_sample.append(torch.LongTensor(T).fill_(i))
		node_offset += O
	# all_rooms_mks = torch.cat(all_rooms_mks, 0)
	all_nodes = torch.cat(all_nodes)
	if len(all_edges) > 0:
		all_edges = torch.cat(all_edges)
	else:
		all_edges = torch.tensor([])       
	all_node_to_sample = torch.cat(all_node_to_sample)
	all_edge_to_sample = torch.cat(all_edge_to_sample)
	return all_nodes, all_edges, all_node_to_sample, all_edge_to_sample


def pad_im(cr_im, final_size=256, bkg_color='white'):    
    new_size = int(np.max([np.max(list(cr_im.size)), final_size]))
    padded_im = Image.new('RGB', (new_size, new_size), 'white')
    padded_im.paste(cr_im, ((new_size-cr_im.size[0])//2, (new_size-cr_im.size[1])//2))
    padded_im = padded_im.resize((final_size, final_size), Image.ANTIALIAS)
    return padded_im

def draw_graph(g_true):
    # build true graph 
    G_true = nx.Graph()
    colors_H = []
    for k, label in enumerate(g_true[0]):
        _type = label+1 
        if _type >= 0:
            G_true.add_nodes_from([(k, {'label':_type})])
            colors_H.append(ID_COLOR[_type])
    for k, m, l in g_true[1]:
        if m > 0:
            G_true.add_edges_from([(k, l)], color='b',weight=4)    
    plt.figure()
    pos = nx.nx_agraph.graphviz_layout(G_true, prog='neato')

    edges = G_true.edges()
    colors = ['black' for u,v in edges]
    weights = [4 for u,v in edges]

    nx.draw(G_true, pos, node_size=1000, node_color=colors_H, font_size=0, font_weight='bold', edge_color=colors, width=weights)
    plt.tight_layout()
    plt.savefig('./dump/_true_graph.jpg', format="jpg")
    rgb_im = Image.open('./dump/_true_graph.jpg')
    rgb_arr = pad_im(rgb_im).convert('RGBA')
    return rgb_arr

        

def draw_masks(masks, real_nodes):

#     transp = Image.new('RGBA', img.size, (0,0,0,0))  # Temp drawing image.
#     draw = ImageDraw.Draw(transp, "RGBA")
#     draw.ellipse(xy, **kwargs)
#     # Alpha composite two images together and replace first with result.
#     img.paste(Image.alpha_composite(img, transp))
    
    bg_img = Image.new("RGBA", (256, 256), (255, 255, 255, 0))  # Semitransparent background.
    for m, nd in zip(masks, real_nodes):
        
        # draw region
        reg = Image.new('RGBA', (32, 32), (0,0,0,0))
        dr_reg = ImageDraw.Draw(reg)
        m[m>0] = 255
        m[m<0] = 0
        m = m.detach().cpu().numpy()
        m = Image.fromarray(m)
        color = ID_COLOR[nd+1]
        r, g, b = webcolors.name_to_rgb(color)
        dr_reg.bitmap((0, 0), m.convert('L'), fill=(r, g, b, 32))
        reg = reg.resize((256, 256))
        
        bg_img.paste(Image.alpha_composite(bg_img, reg))

  
    for m, nd in zip(masks, real_nodes):
        cnt = Image.new('RGBA', (256, 256), (0,0,0,0))
        dr_cnt = ImageDraw.Draw(cnt)
        
        mask = np.zeros((256,256,3)).astype('uint8')
        m[m>0] = 255
        m[m<0] = 0
        m = m.detach().cpu().numpy()[:, :, np.newaxis].astype('uint8')
        m = cv2.resize(m, (256, 256), interpolation = cv2.INTER_AREA) 
        ret,thresh = cv2.threshold(m,127,255,0)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:  
            contours = [c for c in contours]
        color = ID_COLOR[nd+1]
        r, g, b = webcolors.name_to_rgb(color)
        cv2.drawContours(mask, contours, -1, (255, 255, 255), 2)
        
        mask = Image.fromarray(mask)
        dr_cnt.bitmap((0, 0), mask.convert('L'), fill=(r, g, b, 256))
        
        bg_img.paste(Image.alpha_composite(bg_img, cnt))
    
#     im2 = np.zeros((256,256,3)).astype('uint8') + 255
#     for m, nd in zip(masks, real_nodes):
#         m[m>0] = 255
#         m[m<0] = 0
#         m = m.detach().cpu().numpy()[:, :, np.newaxis].astype('uint8')
#         m = cv2.resize(m, (256, 256), interpolation = cv2.INTER_AREA) 
#         ret,thresh = cv2.threshold(m,127,255,0)
#         contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#         if len(contours) > 0:  
#             contours = [c for c in contours]
#         color = ID_COLOR[nd+1]
#         r, g, b = webcolors.name_to_rgb(color)

#         cv2.drawContours(im2, contours, -1, (r, g, b), 2)
#     im2 = Image.fromarray(im2).convert('RGBA')

#     im.paste(im2)
#     out.save('./test.png')
#     im.save('./test_reg.png')

    return bg_img

def draw_floorplan(dwg, junctions, juncs_on, lines_on):

    # draw edges
    for k, l in lines_on:
        x1, y1 = np.array(junctions[k])
        x2, y2 = np.array(junctions[l])
        #fill='rgb({},{},{})'.format(*(np.random.rand(3)*255).astype('int'))
        dwg.add(dwg.line((float(x1), float(y1)), (float(x2), float(y2)), stroke='black', stroke_width=4, opacity=1.0))

    # draw corners
    for j in juncs_on:
        x, y = np.array(junctions[j])
        dwg.add(dwg.circle(center=(float(x), float(y)), r=3, stroke='red', fill='white', stroke_width=2, opacity=1.0))
    return 

