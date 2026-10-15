/*
 * oomtest.c
 *
 * Copyright (C) 2026, bg0d-droid.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 */

/*
 * Allocation-failure test for the TurboJPEG API.  This program replaces
 * malloc(), calloc(), realloc() and free() with an allocator over a private
 * arena that can fail any chosen allocation on demand.  Because the test is
 * linked with the static TurboJPEG library, every allocation the library
 * makes goes through these functions, so each allocation failure point can
 * be tested in turn without any special build options.
 *
 * For every allocation failure point the test checks that the operation
 * either succeeds or fails cleanly and leaves no allocated memory behind.
 * tj3Init() used to leak the instance and the partially-initialized JPEG
 * compression/decompression object when an allocation failed during
 * initialization, and tj3LoadImage() leaked the temporary instance it
 * creates internally.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "turbojpeg.h"


/*
 * A simple free-list allocator over a static arena with coalescing.  It
 * needs no allocation of its own and can serve any number of allocations
 * and frees in any order, so it is safe to use from the replacement
 * allocation functions themselves and from the C library.
 *
 * Every block has a header immediately before its payload; the payload is
 * aligned to the strictest fundamental alignment.  Each block also records
 * whether it was allocated while a test case was running ("scoped"); only
 * scoped blocks take part in the leak check, so one-time lazy allocations
 * made by the C library outside the test cases don't matter.
 */
#define ARENA_SIZE  (8 * 1024 * 1024)

union max_align {
  long double ld;
  void *p;
  long long ll;
};

struct blk {
  struct blk *prev, *next;
  size_t size;
  int is_free;
  int scoped;
  unsigned int magic;
};

#define BLK_MAGIC  0x74757231
#define ALIGN_SIZE  (sizeof(union max_align))
#define HDR_SIZE \
  (((sizeof(struct blk) + ALIGN_SIZE - 1) / ALIGN_SIZE) * ALIGN_SIZE)

static union max_align arena[ARENA_SIZE / sizeof(union max_align)];
#define ARENA_BYTES  (sizeof(arena))

static struct blk *first_blk, *last_blk;
static size_t arena_used;
static long live_scoped;  /* live scoped blocks */
static int arena_error;   /* arena exhaustion or corruption */

/* Allocation failure injection state. */
static int alloc_armed;     /* nonzero while a test case runs */
static long alloc_seq;      /* allocations seen in this run */
static long alloc_fail_at;  /* fail this allocation, 0 = fail none */


/*
 * Round an allocation size up to the arena alignment; returns nonzero on
 * overflow.
 */
static int round_up_size(size_t n, size_t *out)
{
  if (n == 0) n = 1;
  if (n > (size_t)-1 - (ALIGN_SIZE - 1)) return 1;
  *out = ((n + ALIGN_SIZE - 1) / ALIGN_SIZE) * ALIGN_SIZE;
  return 0;
}


static void *arena_alloc(size_t n, int scoped)
{
  struct blk *b;
  size_t need;

  if (round_up_size(n, &need)) {
    arena_error = 1;
    return NULL;
  }

  /* First fit among the free blocks. */
  for (b = first_blk; b != NULL; b = b->next) {
    if (!b->is_free || b->size < need) continue;
    /* Split the block if the remainder can hold a block. */
    if (b->size >= need + HDR_SIZE + ALIGN_SIZE) {
      struct blk *nb;

      nb = (struct blk *)((char *)b + HDR_SIZE + need);
      nb->prev = b;
      nb->next = b->next;
      nb->size = b->size - need - HDR_SIZE;
      nb->is_free = 1;
      nb->scoped = 0;
      nb->magic = 0;
      if (nb->next != NULL) nb->next->prev = nb;
      else last_blk = nb;
      b->next = nb;
      b->size = need;
    }
    b->is_free = 0;
    b->scoped = scoped;
    b->magic = BLK_MAGIC;
    if (scoped) live_scoped++;
    return (char *)b + HDR_SIZE;
  }

  /* Allocate a fresh block at the tail of the arena. */
  if (arena_used + HDR_SIZE + need > ARENA_BYTES) {
    arena_error = 1;
    return NULL;
  }
  b = (struct blk *)((char *)arena + arena_used);
  arena_used += HDR_SIZE + need;
  b->prev = last_blk;
  b->next = NULL;
  b->size = need;
  b->is_free = 0;
  b->scoped = scoped;
  b->magic = BLK_MAGIC;
  if (last_blk != NULL) last_blk->next = b;
  else first_blk = b;
  last_blk = b;
  if (scoped) live_scoped++;
  return (char *)b + HDR_SIZE;
}


/*
 * Release a block and coalesce it with its free neighbours.  A pointer that
 * is not in the arena at all was allocated by some other allocator (parts
 * of the C library allocate internally with symbols this test does not
 * replace); there is no portable way to release such a block here, so it is
 * left alone.
 */
static void arena_free(void *p)
{
  struct blk *b;

  if (p == NULL) return;
  if ((char *)p < (char *)arena + HDR_SIZE ||
      (char *)p >= (char *)arena + ARENA_BYTES)
    return;
  b = (struct blk *)((char *)p - HDR_SIZE);
  if (b->magic != BLK_MAGIC) {
    /* Double free or corruption. */
    arena_error = 1;
    return;
  }
  b->magic = 0;
  b->is_free = 1;
  if (b->scoped) live_scoped--;

  if (b->next != NULL && b->next->is_free) {
    b->size += HDR_SIZE + b->next->size;
    b->next = b->next->next;
    if (b->next != NULL) b->next->prev = b;
    else last_blk = b;
  }
  if (b->prev != NULL && b->prev->is_free) {
    b->prev->size += HDR_SIZE + b->size;
    b->prev->next = b->next;
    if (b->next != NULL) b->next->prev = b->prev;
    else last_blk = b->prev;
  }
}


static void *arena_realloc(void *p, size_t n)
{
  struct blk *b;
  size_t need;
  void *q;

  if (p == NULL) return arena_alloc(n, alloc_armed);
  if (n == 0) {
    arena_free(p);
    return NULL;
  }
  b = (struct blk *)((char *)p - HDR_SIZE);
  if (b->magic != BLK_MAGIC) {
    arena_error = 1;
    return NULL;
  }
  if (round_up_size(n, &need)) {
    arena_error = 1;
    return NULL;
  }
  if (b->size >= need) return p;
  /*
   * Allocate before releasing, so that on failure the original block
   * remains valid and unchanged, as realloc() requires.  The copy
   * inherits the scope of the original block.
   */
  q = arena_alloc(n, b->scoped);
  if (q == NULL) return NULL;
  memcpy(q, p, b->size);
  arena_free(p);
  return q;
}


/*
 * Decide whether this allocation attempt should fail.  The counter runs
 * only while a test case is executing, so allocations made by the C library
 * on behalf of this program itself never fail.
 */
static int should_fail(void)
{
  if (!alloc_armed) return 0;
  alloc_seq++;
  return alloc_fail_at && alloc_seq == alloc_fail_at;
}


void *malloc(size_t n)
{
  if (should_fail()) return NULL;
  return arena_alloc(n, alloc_armed);
}


void *calloc(size_t m, size_t n)
{
  void *p;

  if (should_fail()) return NULL;
  if (m != 0 && n > (size_t)-1 / m) return NULL;
  p = arena_alloc(m * n, alloc_armed);
  if (p != NULL) memset(p, 0, m * n);
  return p;
}


void *realloc(void *p, size_t n)
{
  if (should_fail()) return NULL;
  return arena_realloc(p, n);
}


void free(void *p)
{
  arena_free(p);
}


/* ---- test cases ---- */

#define PPM_FILE  "oomtest_input.ppm"


/*
 * Create a small PPM image for tj3LoadImage() to read.  This also warms up
 * any one-time lazy allocations the C library makes for file I/O, so that
 * they happen outside the measured test windows.
 */
static int write_ppm(void)
{
  FILE *f;
  int x, y;

  if ((f = fopen(PPM_FILE, "wb")) == NULL) return 1;
  fprintf(f, "P6\n%d %d\n255\n", 64, 48);
  for (y = 0; y < 48; y++) {
    for (x = 0; x < 64 * 3; x++) {
      unsigned char v = (unsigned char)(x * 5 + y * 11);

      fwrite(&v, 1, 1, f);
    }
  }
  fclose(f);
  return 0;
}


/*
 * Run one tj3Init() test case.  With fail_at == 0 the initialization must
 * succeed; otherwise an initialization failure is tolerated as long as the
 * instance was cleaned up.  Returns nonzero on a test failure.  If count is
 * not NULL, the number of allocations made is returned there.
 */
static int run_init(int initType, long fail_at, long *count)
{
  tjhandle handle;

  alloc_seq = 0;
  alloc_fail_at = fail_at;
  alloc_armed = 1;
  handle = tj3Init(initType);
  if (handle) tj3Destroy(handle);
  alloc_armed = 0;
  if (count != NULL) *count = alloc_seq;

  if (fail_at == 0 && handle == NULL) {
    fprintf(stderr, "tj3Init(%d) failed without an injected failure\n",
            initType);
    return 1;
  }
  return 0;
}


/*
 * Fail every allocation of tj3Init() in turn and check that no memory
 * leaks in any case.
 */
static int sweep_init(int initType)
{
  long n, i;

  if (run_init(initType, 0, &n)) return 1;
  if (n == 0) {
    fprintf(stderr, "no allocations seen for tj3Init(%d); the allocation "
            "replacements are not being used\n", initType);
    return 1;
  }
  if (live_scoped != 0) {
    fprintf(stderr, "tj3Init(%d) leaked %ld block(s)\n", initType,
            live_scoped);
    return 1;
  }

  for (i = 1; i <= n; i++) {
    if (run_init(initType, i, NULL)) return 1;
    if (live_scoped != 0) {
      fprintf(stderr, "tj3Init(%d) with allocation %ld failing leaked "
              "%ld block(s)\n", initType, i, live_scoped);
      return 1;
    }
    if (arena_error) {
      fprintf(stderr, "tj3Init(%d) with allocation %ld failing hit an "
              "arena error (double free or corruption)\n", initType, i);
      return 1;
    }
  }
  return 0;
}


/*
 * Run one tj3LoadImage() test case.  tj3LoadImage() creates a temporary
 * instance internally, so the initialization of that instance is covered as
 * well.  With fail_at == 0 the load must succeed.
 */
static int run_load(long fail_at, long *count)
{
  tjhandle handle;
  unsigned char *img = NULL;
  int width = 0, height = 0, pixelFormat = TJPF_RGB;

  alloc_seq = 0;
  alloc_fail_at = fail_at;
  alloc_armed = 1;
  handle = tj3Init(TJINIT_COMPRESS);
  if (handle) {
    img = tj3LoadImage8(handle, PPM_FILE, &width, 1, &height, &pixelFormat);
    if (img) tj3Free(img);
    tj3Destroy(handle);
  }
  alloc_armed = 0;
  if (count != NULL) *count = alloc_seq;

  if (fail_at == 0 && (handle == NULL || img == NULL)) {
    fprintf(stderr, "tj3LoadImage() failed without an injected failure\n");
    return 1;
  }
  return 0;
}


static int sweep_load(void)
{
  long n, i;

  if (run_load(0, &n)) return 1;
  if (n == 0) {
    fprintf(stderr, "no allocations seen for tj3LoadImage(); the "
            "allocation replacements are not being used\n");
    return 1;
  }
  if (live_scoped != 0) {
    fprintf(stderr, "tj3LoadImage() leaked %ld block(s)\n", live_scoped);
    return 1;
  }

  for (i = 1; i <= n; i++) {
    if (run_load(i, NULL)) return 1;
    if (live_scoped != 0) {
      fprintf(stderr, "tj3LoadImage() with allocation %ld failing leaked "
              "%ld block(s)\n", i, live_scoped);
      return 1;
    }
    if (arena_error) {
      fprintf(stderr, "tj3LoadImage() with allocation %ld failing hit an "
              "arena error (double free or corruption)\n", i);
      return 1;
    }
  }
  return 0;
}


int main(void)
{
  int ret = 0;

  if (write_ppm()) {
    fprintf(stderr, "Could not create %s\n", PPM_FILE);
    return 1;
  }

  if (sweep_init(TJINIT_COMPRESS) ||
      sweep_init(TJINIT_DECOMPRESS) ||
      sweep_init(TJINIT_TRANSFORM) ||
      sweep_load())
    ret = 1;

  remove(PPM_FILE);
  if (ret == 0) printf("OK\n");
  return ret;
}
