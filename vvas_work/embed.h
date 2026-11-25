#ifndef EMBED_H
#define EMBED_H

/* Size of one face embedding (number of floats) */
#define FACE_EMBEDDING_DIM 512

#ifdef __cplusplus
extern "C" {
#endif

/* Loads the face-recognition model. Returns 0 on success, -1 on failure. */
int face_embedder_init(void);

/*
 * Computes an embedding for one face box.
 * The frame is NV12 (VVAS_VFMT_Y_UV8_420), which is stored in two planes:
 * y_plane:            brightness, 1 byte per pixel
 * uv_plane:           interleaved U and V color values, half as many rows as y_plane
 * frame_width/height: frame size in pixels
 * stride_bytes:       bytes PER ROW of both planes
 * box_x, box_y, box_width, box_height: face box in frame pixels
 * embedding_out:      buffer of FACE_EMBEDDING_DIM floats
 * Returns 0 on success, -1 on failure.
 */
int face_embedder_compute(const unsigned char *y_plane,
                          const unsigned char *uv_plane,
                          int frame_width, int frame_height,
                          int stride_bytes,
                          int box_x, int box_y,
                          int box_width, int box_height,
                          float *embedding_out);

void face_embedder_deinit(void);

#ifdef __cplusplus
}
#endif
#endif
