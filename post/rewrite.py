import sys

filepath = sys.argv[1]
with open(filepath, 'r') as f:
    content = f.read()
    
content = content.replace(
    r"\thm",
    """<div class="block">\n\n<p class="block-title thm">定理</p>"""
).replace(
    r"\def",
    """<div class="block">\n\n<p class="block-title def">定义</p>"""
).replace(
    r"\ok", "</div>"
).replace(
    r"\R", r"\mathbb{R}"
)

with open(filepath, 'w') as f:
    f.write(content)
