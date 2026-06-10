with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()
content = content.replace("use_container_width=True", "use_container_width=True")  # keep as-is for now
with open("app.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Done")
