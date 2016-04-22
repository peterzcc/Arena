import numpy
import os


'''
prepraing data for alov300++
'''
output_path = '/home/sliay/Documents/alov300/alov300_list'
alov_path = '/home/sliay/Documents/alov300/'
video_path = os.path.join(alov_path, 'imagedata++')
roi_path = os.path.join(alov_path, 'alov300++_rectangleAnnotation_full')
categories = sorted(os.listdir(video_path))
alov_list_file = open(os.path.join(output_path, 'alov300.lst'), 'w')

for cat in categories:
    video_per_category = sorted(os.listdir(os.path.join(video_path, cat)))
    gt_per_category = sorted(os.listdir(os.path.join(roi_path, cat)))
    counter = -1
    for gt in gt_per_category:
        counter += 1
        roi = numpy.loadtxt(os.path.join(roi_path, cat, gt))
        index = numpy.int64(roi[:, 0])
        roi = roi[:, [3, 2, 1, 6]]
        roi[:, 2:4] = roi[:, 2:4] - roi[:, 0:2] + 1
        interp_roi = numpy.zeros((index[-1] - index[0] + 1, 4))
        # do linear interpolation
        for i in xrange(4):
            interp_roi[:, i] = numpy.interp(numpy.arange(index[0], index[-1]+1), index, roi[:, i])
        name = gt.split('.')[0]
        image_list_file = open(os.path.join(output_path, name+'.lst'), 'w')
        img_list = sorted(os.listdir(os.path.join(video_path, cat, video_per_category[counter])))
        for i in xrange(interp_roi.shape[0]):
            line = '%s %d %d %d %d\n' % (os.path.join(video_path, cat ,video_per_category[counter], img_list[i+index[0]-1]), interp_roi[i, 0], interp_roi[i, 1], interp_roi[i, 2], interp_roi[i, 3])
            image_list_file.write(line)
        image_list_file.close()
        line = '%s\n' % os.path.join(output_path, name+'.lst')
        alov_list_file.write(line)

alov_list_file.close()



