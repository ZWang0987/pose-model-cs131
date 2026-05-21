# ui interface


import sys


import pygame
pygame.init()




class PoseNetCamera:
 def __init__(self, input_uri="csi://0", network="resnet18-body", threshold=0.15, overlay="links,keypoints"):
   self.available = False
   self.error = None
   self.overlay = overlay
   self.cuda_to_numpy = None


   try:
     import numpy as np
     from jetson_inference import poseNet
     from jetson_utils import videoSource, cudaToNumpy, cudaDeviceSynchronize
   except ImportError as e:
     self.error = f"PoseNet unavailable: {e}"
     return


   self.np = np


   try:
     self.net = poseNet(network, sys.argv, threshold)
     self.input = videoSource(input_uri, argv=sys.argv)
     self.cuda_to_numpy = cudaToNumpy
     self.cuda_device_synchronize = cudaDeviceSynchronize
     self.available = True
   except Exception as e:
     self.error = f"PoseNet init failed: {e}"


 def get_surface(self):
   if not self.available:
     return None, []


   img = self.input.Capture()
   if img is None:
     return None, []


   poses = self.net.Process(img, overlay=self.overlay)
   self.cuda_device_synchronize()
   frame = self.cuda_to_numpy(img)


   if frame.dtype != self.np.uint8:
     frame = self.np.clip(frame, 0, 255).astype(self.np.uint8)


   if frame.ndim == 2:
     frame = self.np.repeat(frame[:, :, None], 3, axis=2)
   elif frame.shape[2] == 4:
     frame = frame[:, :, :3]
   elif frame.shape[2] == 1:
     frame = self.np.repeat(frame, 3, axis=2)
   else:
     frame = frame[:, :, :3]


   frame = self.np.ascontiguousarray(frame)
   surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
   return surface, poses


 def is_streaming(self):
   return self.available and self.input.IsStreaming()




def blit_fit(surface, target_surface, target_rect):
 source_rect = surface.get_rect()
 scale = min(target_rect.width / source_rect.width, target_rect.height / source_rect.height)
 scaled_size = (
   max(1, int(source_rect.width * scale)),
   max(1, int(source_rect.height * scale)),
 )
 scaled = pygame.transform.smoothscale(surface, scaled_size)
 scaled_rect = scaled.get_rect(center=target_rect.center)
 target_surface.blit(scaled, scaled_rect)




def draw_centered_text(surface, text, rect, font, color):
 rendered = font.render(text, True, color)
 text_rect = rendered.get_rect(center=rect.center)
 surface.blit(rendered, text_rect)




#---------------------------------------------
#                  INIT WINDOW
window = pygame.display.set_mode((1000,700))
pygame.display.set_caption("Kareoke + Dance")


#---------------------------------------------
#                  INIT POSE MODEL
pose_camera = PoseNetCamera()


#---------------------------------------------
#                  INIT SONG
SONG_DIR = "songs"
music_loaded = True
try:
 pygame.mixer.music.load(f"{SONG_DIR}/cupid.mp3")
 pygame.mixer.music.play()
except pygame.error as e:
 music_loaded = False
 print(f"Music unavailable: {e}")




#---------------------------------------------
#                  LAYOUT


# constants
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LINE_WIDTH = 2
MARGIN = 20
GAP = 20
LABEL_PADDING = 10


# window labels
label_font = pygame.font.SysFont(None, 26)
status_font = pygame.font.SysFont(None, 22)
left_label = label_font.render("POSE MODEL", True, WHITE)
right_label = label_font.render("NEXT MOVE", True, WHITE)
bottom_label = label_font.render("LYRICS", True, WHITE)


# window heights
window_width, window_height = window.get_size()


top_height = (window_height - (2 * MARGIN) - GAP) // 2
top_width = (window_width - (2 * MARGIN) - GAP) // 2


# create window stamps
left_rect = pygame.Rect(MARGIN, MARGIN, top_width, top_height)
right_rect = pygame.Rect(MARGIN + top_width + GAP, MARGIN, top_width, top_height)
bottom_rect = pygame.Rect(
 MARGIN,
 MARGIN + top_height + GAP,
 window_width - (2 * MARGIN),
 window_height - (2 * MARGIN) - top_height - GAP,
)


# create camera stamp
left_camera_rect = pygame.Rect(
 left_rect.x + LABEL_PADDING,
 left_rect.y + LABEL_PADDING + left_label.get_height() + LABEL_PADDING,
 left_rect.width - (2 * LABEL_PADDING),
 left_rect.height - (3 * LABEL_PADDING) - left_label.get_height(),
)


# create play/pause button (top right window -> top right corner)
button_font = pygame.font.SysFont(None, 24)
button_rect = pygame.Rect(window_width - MARGIN - 100, MARGIN, 100, 40)
button_text = button_font.render("PAUSE", True, BLACK)
is_paused = False


#---------------------------------------------
#                   RUN


# initialize window
run = True


while run:
 pygame.time.delay(15)


 # look to see if window exited
 for e in pygame.event.get():


   """ USER EXIT CALL """
   if e.type == pygame.QUIT:
     run = False


   """ USER SCREEN INPUT """
   if e.type == pygame.MOUSEBUTTONDOWN:
     if button_rect.collidepoint(e.pos): # if user pressed in pause button area (called button_rect)


       is_paused = not is_paused # toggle pause
       if is_paused and music_loaded:
         pygame.mixer.music.pause()
         button_text = button_font.render("PLAY", True, BLACK)
       elif music_loaded:
         pygame.mixer.music.unpause()
         button_text = button_font.render("PAUSE", True, BLACK)
       else:
         button_text = button_font.render("PLAY", True, BLACK) if is_paused else button_font.render("PAUSE", True, BLACK)


 # start new frame
 window.fill(BLACK)


 # draw windows
 pygame.draw.rect(window, WHITE, left_rect, LINE_WIDTH)
 pygame.draw.rect(window, WHITE, right_rect, LINE_WIDTH)
 pygame.draw.rect(window, WHITE, bottom_rect, LINE_WIDTH)


 # draw text
 window.blit(left_label, (left_rect.x + LABEL_PADDING, left_rect.y + LABEL_PADDING))
 window.blit(right_label, (right_rect.x + LABEL_PADDING, right_rect.y + LABEL_PADDING))
 window.blit(bottom_label, (bottom_rect.x + LABEL_PADDING, bottom_rect.y + LABEL_PADDING))


 # draw pose model camera feed
 pose_surface, poses = pose_camera.get_surface()
 if pose_surface is not None:
   blit_fit(pose_surface, window, left_camera_rect)
   pose_count = status_font.render(f"poses: {len(poses)}", True, WHITE)
   window.blit(pose_count, (left_camera_rect.x, left_camera_rect.bottom - pose_count.get_height()))
 elif pose_camera.error:
   draw_centered_text(window, pose_camera.error, left_camera_rect, status_font, WHITE)
 else:
   draw_centered_text(window, "Waiting for pose camera...", left_camera_rect, status_font, WHITE)


 # draw play/pause button
 pygame.draw.rect(window, WHITE, button_rect)
 window.blit(button_text, (button_rect.x + 15, button_rect.y + 10))


 # update to new frame
 pygame.display.flip()


 if pose_camera.available and not pose_camera.is_streaming():
   run = False


# exit pygame
pygame.quit()