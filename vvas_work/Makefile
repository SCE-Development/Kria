CC       = gcc
CXX      = g++
PKGS     = gstreamer-1.0 gstreamer-video-1.0 jansson
CFLAGS   = -fPIC -O2 -Wall $(shell pkg-config --cflags $(PKGS))
CXXFLAGS = -fPIC -O2 -Wall -std=c++17 $(shell pkg-config --cflags opencv4)
LIBS     = $(shell pkg-config --libs $(PKGS)) -lopencv_core -lopencv_imgproc
EXTRA_LIBS ?= -lgstvvasinfermeta-2.0 \
              -lvitis_ai_library-facefeature -lvitis_ai_library-model_config \
              -lvitis_ai_library-dpu_task -lvitis_ai_library-math
TARGET   = libvvas_results.so
OBJS     = vvas_results.o embed.o

$(TARGET): $(OBJS)
	$(CXX) -shared -o $@ $(OBJS) $(LIBS) $(EXTRA_LIBS)

vvas_results.o: vvas_results.c embed.h
	$(CC) $(CFLAGS) -c $< -o $@

embed.o: embed.cpp embed.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)