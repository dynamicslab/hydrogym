/* Print the compute capability of the visible NVIDIA GPUs as ccXY (e.g. cc89),
 * through the CUDA driver API, so CUDA_VISIBLE_DEVICES is honoured. Used by
 * the maia wrapper of the CUDA release image. Exit status 1: no driver
 * (libcuda cannot be loaded); 2: the visible GPUs have different compute
 * capabilities (then all of them are printed); 3: the driver sees no GPU
 * (none visible, e.g. CUDA_VISIBLE_DEVICES empty). Built without CUDA headers: libcuda is opened at run
 * time, it comes from the host driver. */
#include <dlfcn.h>
#include <stdio.h>

typedef int (*init_t)(unsigned int);
typedef int (*count_t)(int *);
typedef int (*get_t)(int *, int);
typedef int (*attr_t)(int *, int, int);

int main(void) {
    void *h = dlopen("libcuda.so.1", RTLD_NOW);
    if (!h) return 1;
    init_t cuInit = (init_t)dlsym(h, "cuInit");
    count_t cuDeviceGetCount = (count_t)dlsym(h, "cuDeviceGetCount");
    get_t cuDeviceGet = (get_t)dlsym(h, "cuDeviceGet");
    attr_t cuDeviceGetAttribute = (attr_t)dlsym(h, "cuDeviceGetAttribute");
    if (!cuInit || !cuDeviceGetCount || !cuDeviceGet || !cuDeviceGetAttribute) return 1;
    int n = 0;
    if (cuInit(0) != 0 || cuDeviceGetCount(&n) != 0 || n < 1) return 3;
    int first = -1, mixed = 0;
    char out[1024] = "";
    int len = 0;
    for (int i = 0; i < n; i++) {
        int dev, major = 0, minor = 0;
        if (cuDeviceGet(&dev, i) != 0) return 1;
        cuDeviceGetAttribute(&major, 75, dev); /* CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR */
        cuDeviceGetAttribute(&minor, 76, dev); /* CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR */
        int cc = 10 * major + minor;
        if (first < 0) first = cc;
        else if (cc != first) mixed = 1;
        if (len < (int)sizeof(out) - 16) len += snprintf(out + len, sizeof(out) - len, "%scc%d", i ? " " : "", cc);
    }
    if (mixed) { printf("%s\n", out); return 2; }
    printf("cc%d\n", first);
    return 0;
}
