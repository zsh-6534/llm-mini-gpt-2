import re
import json
import ftfy


def format_text(input_path, output_path="src/dataset/demo-小说.txt"):

    chapter_pattern = re.compile(
        r'^(?:第[0-9一二三四五六七八九十百千]+卷\s+[^\s]+\s+)?'
        r'第([0-9一二三四五六七八九十百千]+)章\s*(.+)'
    )  # 新增章节识别
    clean_pattern = re.compile(
        r'\(看小说到网\).*|'
        r'\(未完待续.*?\)|'
        r'PS：.*|'
        r'------------|'
        r'^[\s\u3000]+|'
        r'[\s\u3000]+$|'
        r'（.*?）|'
        r'\(.*?\)'
    )

    current_title = "默认标题"
    content = []

    with open(input_path, 'r', encoding='utf-8') as f_in, \
            open(output_path, 'a', encoding='utf-8') as f_txt:

        for line in f_in:
            # 清理和修复文本
            line = ftfy.fix_text(line.strip())
            line = clean_pattern.sub('', line)

            if not line:
                continue

            # 章节识别
            if chapter_match := chapter_pattern.match(line):
                if content:
                    # 写入文本文件
                    f_txt.write('\n'.join(content))
                    content = []
                current_title = f"第{chapter_match.group(1)}章 {chapter_match.group(2)}"

            else:
                content.append(line)

        # 写入最后章节
        if content:
            f_txt.write('\n'.join(content))


# 清洗数据 - 自己随便整个文本都行
format_text("src/dataset/source/凡人修仙传.txt", "src/dataset/target/凡人修仙传.txt")
