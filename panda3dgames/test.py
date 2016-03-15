from math import pi, sin, cos
from panda3d.core import loadPrcFileData

loadPrcFileData("",
                """
   load-display p3tinydisplay # to force CPU only rendering (to make it available as an option if everything else fail, use aux-display p3tinydisplay)
   window-type offscreen # Spawn an offscreen buffer (use window-type none if you don't need any rendering)
   audio-library-name null # Prevent ALSA errors
   show-frame-rate-meter 0
   undecorated 1
   sync-video 0
   clock-mode limited
   clock-frame-rate 300
""")
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from direct.actor.Actor import Actor
from direct.interval.IntervalGlobal import Sequence
from panda3d.core import GraphicsOutput
from panda3d.core import FrameBufferProperties
from panda3d.core import Point2, Point3
from panda3d.core import Texture, WindowProperties, GraphicsPipe, PNMImage
import numpy
import cv2
from PIL import Image, ImageDraw


def compute2d_pos(nodePath, point=Point3(0, 0, 0), none_on_outof_view=True):
    """ Computes a 3-d point, relative to the indicated node, into a
    2-d point as seen by the camera.  The range of the returned value
    is based on the len's current film size and film offset, which is
    (-1 .. 1) by default. """

    # Convert the point into the camera's coordinate space
    p3d = base.cam.getRelativePoint(nodePath, point)
    # Ask the lens to project the 3-d point to 2-d.
    p2d = Point2()
    inview = base.camLens.project(p3d, p2d)
    if none_on_outof_view:
        if inview:
            return p2d
        else:
            return None
    else:
        return p2d


# TODO Use cv2.minAreaRect
def get_screen_bb(render, nodePath, type='simple'):
    minPos, maxPos = nodePath.getTightBounds()
    pt2d = numpy.zeros((8, 2), dtype=numpy.float32)
    x = (minPos[0], maxPos[0])
    y = (minPos[1], maxPos[1])
    z = (minPos[2], maxPos[2])
    for i in range(2):
        for j in range(2):
            for k in range(2):
                pt2d[i * 4 + 2 * j + k, :] = \
                    compute2d_pos(nodePath=render, point=Point3(x[i], y[j], z[k]),
                                  none_on_outof_view=False)
    if type == 'simple':
        minX = max(pt2d[:, 0].min(), -1)
        minY = max(pt2d[:, 1].min(), -1)
        maxX = min(pt2d[:, 0].max(), 1)
        maxY = min(pt2d[:, 1].max(), 1)
        siz = (maxX - minX, maxY - minY)
        center = ((maxX + minX) / 2, (maxY + minY) / 2)
        if siz[0] <= 0 or siz[1] <= 0:
            return None
        else:
            return ((center[0] + 1)/2, (center[1] + 1)/2, siz[0]/2, siz[1]/2)
    return None

class MyApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self, windowType='offscreen')
        self.tex = Texture('tex')
        base.win.addRenderTexture(self.tex, GraphicsOutput.RTMCopyRam)
        self.tex.setClearColor((0, 0, 0, 1))
        self.tex.clearImage()
        self.screen_buffer = numpy.empty((3, self.tex.getXSize(), self.tex.getYSize()),
                                         dtype='uint8')

        self.scene = self.loader.loadModel('environment')
        self.scene.reparentTo(self.render)
        self.scene.setScale(0.25, 0.25, 0.25)
        self.scene.setPos(-8, 42, 0)
        self.taskMgr.add(self.spinCameraTask, "SpinCameraTask")

        self.pandaActor = Actor("panda-model",
                                {"walk": "panda-walk4"})
        self.pandaActor.setScale(0.005, 0.005, 0.005)
        self.pandaActor.reparentTo(self.render)
        self.pandaActor.loop("walk")
        pandaPosInterval1 = self.pandaActor.posInterval(10,
                                                        Point3(0, -10, 0),
                                                        startPos=Point3(0, 10, 0))
        pandaPosInterval2 = self.pandaActor.posInterval(10,
                                                        Point3(0, 10, 0),
                                                        startPos=Point3(0, -10, 0))
        pandaHprInterval1 = self.pandaActor.hprInterval(3,
                                                        Point3(180, 0, 0),
                                                        startHpr=Point3(0, 0, 0))
        pandaHprInterval2 = self.pandaActor.hprInterval(3,
                                                        Point3(0, 0, 0),
                                                        startHpr=Point3(180, 0, 0))

        # Create and play the sequence that coordinates the intervals.
        self.pandaPace = Sequence(pandaPosInterval1,
                                  pandaHprInterval1,
                                  pandaPosInterval2,
                                  pandaHprInterval2,
                                  name="pandaPace")
        self.pandaPace.loop()

    def spinCameraTask(self, task):
        print task.frame
        angleDegrees = task.frame/1000 * 6.0
        angleRadians = angleDegrees * (pi / 180.0)
        self.camera.setPos(20 * sin(angleRadians), -20.0 * cos(angleRadians), 3)
        self.camera.setHpr(angleDegrees, 0, 0)
        base.graphicsEngine.renderFrame()
        img = self.tex.getRamImageAs("RGB")
        if img:
            print img, self.tex.getXSize(), self.tex.getYSize()
            pos = get_screen_bb(self.render, self.pandaActor)
            if pos is not None:
                cx = (1 - pos[0]) * self.tex.getXSize()
                cy = (1 - pos[1]) * self.tex.getYSize()
                sx = pos[2] * self.tex.getXSize()
                sy = pos[3] * self.tex.getYSize()
            image = Image.frombytes(
                mode="RGB", size=(self.tex.getXSize(), self.tex.getYSize()), data=img.getData())
            image = image.rotate(180)
            if pos is not None:
                draw = ImageDraw.Draw(image)
                draw.rectangle([cx -sx/2, cy - sy/2, cx + sx/2 , cy + sy/2])
            image.show()
            ch = raw_input()
        return Task.cont


app = MyApp()
app.run()
