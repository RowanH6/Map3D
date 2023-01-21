import cv2 as cv
import numpy as np
import google_streetview.api as gsv
import matplotlib.pyplot as plt
import math

# enter venv : > src/Scripts/activate

# feature match params

sift = cv.SIFT_create()

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv.FlannBasedMatcher(index_params, search_params)

# get key

f = open('MapRH/key.txt', 'r')
API_KEY = f.read()

# find near addresses

def iter_string(string, p):
	fstword = string.split()[0]

	remaining = ""

	for i in range (1, len(string.split())):
		remaining = remaining + " " + string.split()[i]

	fstword = str(int(fstword) + p)

	reconst = fstword + remaining

	return reconst

# get the coordinates of the images

def get_c(r1, r2):
	k = ['date', 'location', 'pano_id', 'status']

	for i, kv in enumerate(r1.metadata[:10]):
		for ki in k:
			if ki in kv:
				if ki == 'location':
					cl1 = [kv[ki]['lng'], kv[ki]['lat']]

	for i, kv in enumerate(r2.metadata[:10]):
		for ki in k:
			if ki in kv:
				if ki == 'location':
					cl2 = [kv[ki]['lng'], kv[ki]['lat']]

	return cl1, cl2

# calculate distance between coordinates

def c_dist(c1, c2):
	long1, lat1 = c1[0], c1[1]
	long2, lat2 = c2[0], c2[1]

	R = 6317

	phi1 = lat1 * math.pi / 180
	phi2 = lat2 * math.pi / 180
	d_phi = (lat2 - lat1) * math.pi / 180
	d_lam = (long2 - long1) * math.pi / 180

	a = math.sin(d_phi/2) * math.sin(d_phi/2) + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam/2) * math.sin(d_lam/2)
	c = 2*math.atan2(math.sqrt(a), math.sqrt(1 - a))
	d = R * c * 1000

	return d


# get single image

def get_image(loc, heading, pitch, name):

	params = [{
		'size': '480x480', # max 640x640 pixels
		'location': loc,
		'heading': heading,
		'pitch': pitch,
		'key': API_KEY 
	}]
	
	results = gsv.results(params)

	for i, url in enumerate(results.links):
		if results.metadata[i]['status'] == 'OK':
			file_path = 'MapRH/Images/' + name + '.jpg'
			gsv.helpers.download(url, file_path)

	return results

# get all images

def get_scene(loc):
	res = []

	locs = [iter_string(loc, 2), iter_string(loc, 4)]
	count = 0

	for l in locs:
		res.append(get_image(l, 157, 0, str(count)))
		count = count + 1

	c1, c2 = get_c(res[0], res[1])
	return c_dist(c1, c2)

def draw_pts(img, pts):
	img_out = img
	for pt in pts:
		x = pt[0]
		y = pt[1]

		img_out = cv.circle(img_out, (int(x), int(y)), radius=5, color=(0, 255, 0), thickness=3)
	
	return img_out

# pixel disparity

def disp(x1, x2):
    return abs(x1 - x2)

# 3D coordinate calculation

def pt(pL, pR):
    u_l, v_l = pL[0], pL[1]
    u_r, v_r = pR[0], pR[1]

    d = disp(u_l, u_r)

    x = (b/d) * (u_l - im1.shape[0]/2)
    y = (b/d)  * (v_l - im1.shape[1]/2)
    z = (b/d) * 100

    return [x, y, z]

b = get_scene('ENTER ADDRESS HERE') #c_dist([43.6684822,-79.3950423], [43.6685049,-79.394918])

# load images and detect matching points

im1 = cv.imread('MapRH/Images/0.jpg')
im2 = cv.imread('MapRH/Images/1.jpg')

pts_1, desc_1 = sift.detectAndCompute(im1, None)
pts_2, desc_2 = sift.detectAndCompute(im2, None)

matches = flann.knnMatch(desc_1, desc_2, k=2)

masked_1_pts = []
masked_2_pts = []

lowe_coeff = 0.8

for i0, (m0,n0) in enumerate(matches):
    if m0.distance < lowe_coeff * n0.distance:
        masked_2_pts.append(pts_2[m0.trainIdx].pt)
        masked_1_pts.append(pts_1[m0.queryIdx].pt)

masked_1_pts = np.int32(masked_1_pts)
masked_2_pts = np.int32(masked_2_pts)

F, mask = cv.findFundamentalMat(masked_1_pts, masked_2_pts, cv.FM_LMEDS)

masked_1_pts = masked_1_pts[mask.ravel()==1]
masked_2_pts = masked_2_pts[mask.ravel()==1]

# calculate x, y, z values

pts_3D = []

for p in range (len(masked_1_pts[:, 0])):
    pt_3D = pt(masked_1_pts[p, :], masked_2_pts[p, :])
    pts_3D.append(pt_3D)

pts_3D = np.double(pts_3D)

# plot on x, y, z axes

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.scatter3D(pts_3D[:, 0], pts_3D[:, 1], pts_3D[:, 2], 'gray')
# ax.set_zlim(0, max(pts_3D[:, 2]) + 0.2)
# ax.set_ylim(min(pts_3D[:, 1]) - 0.1, max(pts_3D[:, 1]) + 0.1)
# ax.set_xlim(min(pts_3D[:, 0]) - 0.1, max(pts_3D[:, 0]) + 0.1)

plt.show()

# pt_im1 = draw_pts(im1, masked_1_pts)
# pt_im2 = draw_pts(im2, masked_2_pts)

# plt.imshow(pt_im2)
# plt.show()