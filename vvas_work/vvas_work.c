#include <stdio.h>
#include <stdlib.h>

#include <gst/gst.h>
#include <jansson.h>
#include <vvas/vvas_kernel.h>
#include <gst/vvas/gstinferencemeta.h>
#include "embed.h"

typedef struct {
  FILE *output_file;
  int frame_count;
} PluginContext;

/*This can be optional if we don't want to write this data to a file*/
// Writes each detected face's bounding box (x,y,w,h) as JSON to the file. 
// The Argumnets take the file to wrote to, a prediction node, and the number of faces already written
// Note: All fprintf() are being printed into the notebook
static int
write_faces (FILE *output_stream, GstInferencePrediction *parent_prediction,
    int faces_written)
{
  //returns a list of the child predictions under that node and each child is one detected face
  GSList *child_predictions =
      gst_inference_prediction_get_children (parent_prediction);
  GSList *list_item;

  for (list_item = child_predictions; list_item; list_item = list_item->next) {
    GstInferencePrediction *face_prediction =
        (GstInferencePrediction *) list_item->data;
    fprintf (output_stream, "%s{\"x\":%d,\"y\":%d,\"w\":%u,\"h\":%u}",
        faces_written ? "," : "",
        face_prediction->bbox.x, face_prediction->bbox.y,
        face_prediction->bbox.width, face_prediction->bbox.height);

    // count the face, then visit nested predictions (cascade models)
    faces_written++;
    faces_written = write_faces (output_stream, face_prediction, faces_written);
  }

  g_slist_free (child_predictions);
  return faces_written;
}

int32_t xlnx_kernel_init (VVASKernel *handle) {
  const char *output_path = "/tmp/vvas_results.jsonl";

  // Look into the config block of results.json; if it holds a string, it replaces the default
  json_t *output_file_setting = json_object_get(handle->kernel_config, "output-file");
  if(json_is_string(output_file_setting)) {
    output_path = json_string_value(output_file_setting);
  }

  PluginContext *context = (PluginContext *) calloc (1, sizeof(PluginContext));
  context->output_file = fopen(output_path, "w"); // Where "w" empties the file if it exists 
  if(!context->output_file) {
    fprintf(stderr, "vvas_results: cannot open %s\n", output_path); // Replace null-terminated string (char *) with output_path
    free(context);
    return -1;
  }

  // If the model fails to load, box are still written, just without the embeddings
  if(face_embedder_init() != 0) {
    fprintf(stderr, "vvas_results: face embedder init failed\n");
  }

  handle->kernel_priv=context;
  fprintf(stderr, "vvas_results: init good, writing to %s\n", output_path);
  return 0;
}

// Runs for every frame. Writes one JSON line wiht the frame number, frame size, and each face's box. 
// This talks to a embedding model, where every 10th frame that contains a face, it computes an embedding for the first
// face and appends it to /tmp/embeddings.jsonl
int32_t xlnx_kernel_start (VVASKernel *handle, int32_t start, VVASFrame *input[MAX_NUM_OBJECT], 
  VVASFrame *output[MAX_NUM_OBJECT]
) {

  PluginContext *context = (PluginContext *) handle->kernel_priv;
  VVASFrame *video_frame = input[0];
  GstBuffer *gst_buffer = (GstBuffer *) video_frame->app_priv;
  GstInferenceMeta *inference_meta = gst_buffer ? 
    (GstInferenceMeta *) gst_buffer_get_meta(gst_buffer, gst_inference_meta_api_get_type()) : NULL;
  
  // inference_meta and inference_meta->prediction are both pointers 
  // inside a logical condtion, it is the same as if we are checking with "!= NULL"
  // If the pointer is not null, it is treated as True and is treated as a "1"
  int has_predictions = (inference_meta && inference_meta->prediction);

  fprintf (context->output_file,
      "{\"frame\":%d,\"width\":%u,\"height\":%u,\"has_meta\":%s,\"faces\":[",
      ++context->frame_count, video_frame->props.width, video_frame->props.height,
      has_predictions ? "true" : "false");

  if (has_predictions)
    write_faces (context->output_file, inference_meta->prediction, 0);

  fprintf (context->output_file, "]}\n");
  fflush (context->output_file);

  static int frames_with_faces = 0;
  if (has_predictions) {
    GSList *face_list =
        gst_inference_prediction_get_children (inference_meta->prediction);

    if (face_list && (frames_with_faces++ % 10) == 0) {
      GstInferencePrediction *first_face = (GstInferencePrediction *) face_list->data;
      float face_embedding[FACE_EMBEDDING_DIM];

      /* The frame is NV12: a Y plane followed by an interleaved UV plane */
      int embed_status = face_embedder_compute (
          (unsigned char *) video_frame->vaddr[0],
          (unsigned char *) video_frame->vaddr[1],
          video_frame->props.width, video_frame->props.height,
          video_frame->props.stride,
          first_face->bbox.x, first_face->bbox.y,
          first_face->bbox.width, first_face->bbox.height,
          face_embedding);

      if (embed_status == 0) {
        /* Save the full normalized embedding, one JSON line per face */
        FILE *embeddings_file = fopen ("/tmp/embeddings.jsonl", "a");
        if (embeddings_file) {
          fprintf (embeddings_file, "{\"frame\":%d,\"embedding\":[",
              context->frame_count);
          for (int value_index = 0; value_index < FACE_EMBEDDING_DIM; value_index++)
            fprintf (embeddings_file, "%s%.6f",
                value_index ? "," : "", face_embedding[value_index]);
          fprintf (embeddings_file, "]}\n");
          fclose (embeddings_file);
        }
      } else {
        fprintf (stderr, "vvas_results: embedding failed\n");
      }
    }
    g_slist_free (face_list);
  }

  return 0;
}

/*
  vvas_xfilter calls it right after start. Once work is finished inside start, it just returns 0.
  This has to exist because vvas_xfilter will not run/load the library missing any of the four functions 
*/
int32_t xlnx_kernel_done (VVASKernel *handle) { return 0; }

/*
  Runs when the pipeline stops.
  Closes the output file, frees the context and releases the face model.
*/
int32_t
xlnx_kernel_deinit (VVASKernel *handle)
{
  PluginContext *context = (PluginContext *) handle->kernel_priv;
  if (context) {
    if (context->output_file) fclose (context->output_file);
    free (context);
  }
  face_embedder_deinit ();
  return 0;
}