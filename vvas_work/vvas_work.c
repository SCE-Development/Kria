#include <stdio.h>
#include <stdlib.h>

//Will show error in a text editor
#include <gst/gst.h>
#include <jansson.h>
#include <vvas/vvas_kernel.h>
#include <gst/vvas/gstinferencemeta.h>

// Create Ctx struct 
// File *fp is the open output file, /tmp/vvas_results.jsonl
// int frame is a counter for how many frames the plugin has seen when we run it
typedef struct { 
  FILE *fp; 
  int frame; 
} Ctx;

/*
  Prints the face for one prediction. 
  The arguments takes the file to write to, a prediction node, and n, the number of faces already written
  Writes each detected face's bounding box (x,y,w,h) as JSON to the file.
*/
static int
write_faces (FILE *fp, GstInferencePrediction *pred, int n)
{
  //returns a list of the child predictions under that node and each child is one detected face
  GSList *kids = gst_inference_prediction_get_children (pred);
  GSList *it;

  for (it = kids; it; it = it->next) {
    GstInferencePrediction *p = (GstInferencePrediction *) it->data;
    fprintf (fp, "%s{\"x\":%d,\"y\":%d,\"w\":%u,\"h\":%u}",
        n ? "," : "", p->bbox.x, p->bbox.y, p->bbox.width, p->bbox.height);

    // n++ counts the face seen and then it recursively calls itself on the child. 
    // For just a single face, the recursive call does nothing
    n++;
    n = write_faces (fp, p, n);   /* cascade models: visit nested predictions */
  }

  //frees the list container 
  g_slist_free (kids);
  return n;
}


/*
  Runs once when the pipeline starts
  Reads the output file path from the JSON config, opens that file, and stores it in handle->keren_priv
  so that later functions can use it.
*/
int32_t
xlnx_kernel_init (VVASKernel *handle)
{
  const char *path = "/tmp/vvas_results.jsonl";

  // looks in the config block of results.json 
  // If it holds a string, it replaces the default
  json_t *v = json_object_get (handle->kernel_config, "output-file"); 
  if (json_is_string (v)) 
    path = json_string_value (v);

  Ctx *c = (Ctx *) calloc (1, sizeof (Ctx));
  c->fp = fopen (path, "w"); // Opens the file for writing, and the "w" empties the file if it exists, if not then it creates it
  if (!c->fp) {
    fprintf (stderr, "vvas_results: cannot open %s\n", path);
    free (c);
    return -1;
  }
  handle->kernel_priv = c;
  fprintf (stderr, "vvas_results: init ok, writing to %s\n", path);
  return 0;
}


/*
  Runs for every frame
  Finds the metadata on the frame's GstBuffer and writes one JSON line that shows the 
  frame number, frame size, and the bounding box of each detected face.

  Buffer - Specific block of memory allocated in VRAM to store temporary raw data
*/
int32_t
xlnx_kernel_start (VVASKernel *handle, int32_t start,
    VVASFrame *input[MAX_NUM_OBJECT], VVASFrame *output[MAX_NUM_OBJECT])
{
  Ctx *c = (Ctx *) handle->kernel_priv;
  GstBuffer *buf = (GstBuffer *) input[0]->app_priv;
  GstInferenceMeta *meta = buf ? (GstInferenceMeta *)
      gst_buffer_get_meta (buf, gst_inference_meta_api_get_type ()) : NULL;
  int has = (meta && meta->prediction);

  fprintf (c->fp, "{\"frame\":%d,\"width\":%u,\"height\":%u,\"has_meta\":%s,\"faces\":[",
      ++c->frame, input[0]->props.width, input[0]->props.height,
      has ? "true" : "false");

  if (has)
    write_faces (c->fp, meta->prediction, 0);

  fprintf (c->fp, "]}\n");
  fflush (c->fp);
  return 0;
}


/*
  vvas_xfilter calls it right after start. Once work is finished inside start, it just returns 0.
  This has to exist because vvas_xfilter will not run/load the library missing any of the four functions 
*/
int32_t xlnx_kernel_done (VVASKernel *handle) { return 0; }

/*
  Runs when the pipeline stops
  Closes the the File *fp and frees the structs 
*/
int32_t
xlnx_kernel_deinit (VVASKernel *handle)
{
  Ctx *c = (Ctx *) handle->kernel_priv;
  if (c) { if (c->fp) fclose (c->fp); free (c); }
  return 0;
}