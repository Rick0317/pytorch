load("@fbcode_macros//build_defs:python_pytest.bzl", "python_pytest")

def functorch_test(name, deps = None, use_gpu = False):
    test_name = "test_{}".format(name)
    src = "test_{}.py".format(name)

    deps = deps or []
    deps.extend([
        "//caffe2/functorch/test:test-lib",
        "//python/wheel/pytest:pytest",
    ])

    tags = ["run_as_bundle"]

    if use_gpu:
        tags.extend([
            "re_opts_capabilities={\"platform\": \"gpu-remote-execution\", \"subplatform\": \"P100\"}",
            "supports_remote_execution",
            "tpx:experimental-shard-size-for-bundle=100",
        ])

    python_pytest(
        name = test_name,
        srcs = [src],
        base_module = "",
        tags = tags,
        deps = deps,
        compile = "with-source",
    )
