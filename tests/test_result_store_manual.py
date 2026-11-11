
import json

from app.storage.result_store import ResultStore


def main():
    # 模拟三条执行结果，先测试存储功能本身
    results = [
        {
            "sample_id": "sample_0001",
            "question": "消防法第六十条规定了什么？",
            "ground_truth": "依法处罚",
            "answer": "根据消防法第六十条依法处罚。",
            "success": True,
            "latency_ms": 1200.5,
            "error": None,
        },
        {
            "sample_id": "sample_0002",
            "question": "消防安全重点单位有哪些责任？",
            "ground_truth": "落实消防安全责任制",
            "answer": "落实消防安全责任制。",
            "success": True,
            "latency_ms": 1500.2,
            "error": None,
        },
    ]

    store = ResultStore(output_dir="results")

    output_path = store.save_results(results)

    print("\n========== 结果存储测试 ==========")
    print("保存路径：", output_path)
    print("文件存在：", output_path.exists())

    with output_path.open("r", encoding="utf-8") as file:
        saved_records = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    print("保存记录数：", len(saved_records))
    print("第一条样本 ID：", saved_records[0]["sample_id"])
    print("运行 ID：", saved_records[0]["run_id"])
    print("==================================")


if __name__ == "__main__":
    main()
