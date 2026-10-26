#include <stdio.h>
#include <stdlib.h>

//Will show error in a text editor
#include <gst/gst.h>
#include <jansson.h>
#include <vvas/vvas_kernel.h>
#include <gst/vvas/gstinferencemeta.h>

typedef struct { FILE *fp; int frame; } Ctx;

static int
write_faces (FILE *fp, GstInferencePrediction *pred, int n)
{
  GSList *kids = gst_inference_prediction_get_children (pred);
  GSList *it;

  for (it = kids; it; it = it->next) {
    GstInferencePrediction *p = (GstInferencePrediction *) it->data;
    fprintf (fp, "%s{\"x\":%d,\"y\":%d,\"w\":%u,\"h\":%u}",
        n ? "," : "", p->bbox.x, p->bbox.y, p->bbox.width, p->bbox.height);
    n++;
    n = write_faces (fp, p, n);   /* cascade models: visit nested predictions */
  }
  g_slist_free (kids);
  return n;
}

int32_t
xlnx_kernel_init (VVASKernel *handle)
{
  const char *path = "/tmp/vvas_results.jsonl";
  json_t *v = json_object_get (handle->kernel_config, "output-file");
  if (json_is_string (v))
    path = json_string_value (v);

  Ctx *c = (Ctx *) calloc (1, sizeof (Ctx));
  c->fp = fopen (path, "w");
  if (!c->fp) {
    fprintf (stderr, "vvas_results: cannot open %s\n", path);
    free (c);
    return -1;
  }
  handle->kernel_priv = c;
  fprintf (stderr, "vvas_results: init ok, writing to %s\n", path);
  return 0;
}

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

int32_t xlnx_kernel_done (VVASKernel *handle) { return 0; }

int32_t
xlnx_kernel_deinit (VVASKernel *handle)
{
  Ctx *c = (Ctx *) handle->kernel_priv;
  if (c) { if (c->fp) fclose (c->fp); free (c); }
  return 0;
}