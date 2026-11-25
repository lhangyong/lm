"""在装有 rknn-toolkit2 的 Linux 机器上，把 ONNX 转成 INT8 的 .rknn。

为什么不在这台 Windows 上转：
  RKNN 官方工具链 rknn-toolkit2 只提供 Linux 包（Ubuntu），本机未安装，
  所以这里先交付 float32 ONNX + 校准图；到 Linux 工具链机器上跑本脚本完成转换和 INT8 量化。

依赖：
  rknn-toolkit2（版本要与板子上的 NPU 驱动/rknn_runtime 对应），Python 3.8 或 3.10。

用法（在本部署包根目录执行）：
  python convert_rknn.py --platform rk3588
  # RK3568 板子用：python convert_rknn.py --platform rk3568

产出：
  metal_nut_patchcore_backbone.rknn  —— 拷到板子上，配合 memory_bank.npz 使用
"""
import argparse

from rknn.api import RKNN

ONNX_MODEL = "metal_nut_patchcore_backbone.onnx"
CALIB_LIST = "calib_list.txt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", default="rk3588", choices=["rk3588", "rk3568"],
                    help="目标板子平台")
    ap.add_argument("--output", default="metal_nut_patchcore_backbone.rknn",
                    help="输出 rknn 文件名")
    args = ap.parse_args()

    rknn = RKNN(verbose=True)

    # 预处理口径（与训练/原工程一致）：
    #   读图 RGB，像素值除以 255 归一化到 0~1；不做额外减均值。
    #   RKNN 里用 mean=0、std=255 表达这一归一化。
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform=args.platform,
    )

    print("--> 加载 ONNX:", ONNX_MODEL)
    rknn.load_onnx(model=ONNX_MODEL)

    print("--> 构建并 INT8 量化（校准集：calib_list.txt 里的正常件）")
    rknn.build(do_quantization=True, dataset=CALIB_LIST)

    print("--> 导出 RKNN:", args.output)
    rknn.export_rknn(args.output)
    rknn.release()
    print("完成:", args.output)


if __name__ == "__main__":
    main()
