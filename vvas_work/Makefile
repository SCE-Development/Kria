
CC = gcc
PKGS = gstreamer-1.0 gstreamer-video-1.0 jansson
CFLAGS = -shared -fPIC -02 Wall CC      = gcc
PKGS    = gstreamer-1.0 gstreamer-video-1.0 jansson
CFLAGS  = -shared -fPIC -O2 -Wall $(shell pkg-config --cflags $(PKGS))
LIBS    = $(shell pkg-config --libs $(PKGS))
TARGET  = libvvas_results.so

$(TARGET): vvas_results.c
	$(CC) $(CFLAGS) $(EXTRA_INC) $< -o $@ $(LIBS) $(EXTRA_LIBS)

clean:
	rm -f $(TARGET)