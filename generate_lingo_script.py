from mako.template import Template

# 定义模板文件路径和生成的 Lingo 脚本文件路径
template_path = 'example_template.mako'
output_script_path = 'generated_script.lng'

# 定义模板变量
model_name = 'ExampleModel'
variables = 'X, Y, Z'
objective_function = '2*X + 3*Y - Z'
constraints = 'X + Y >= 10\nX - Z <= 5'

# 从模板文件渲染模板并生成 Lingo 脚本
template = Template(filename=template_path)
output_script = template.render(
    model_name=model_name,
    variables=variables,
    objective_function=objective_function,
    constraints=constraints
)

print(output_script)
# 将生成的脚本写入文件
with open(output_script_path, 'w') as f:
    f.write(output_script)


