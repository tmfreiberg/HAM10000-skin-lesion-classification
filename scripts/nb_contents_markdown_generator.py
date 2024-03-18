def make_anchor(heading, no_chars = 10):
    head_list = heading.replace(".", "").replace(",", "").replace("(","").replace(")","").replace("/","").replace("&", "").lower()
    head_list = head_list.split(' ')
    anchor_id = ''
    i = 0
    while len(anchor_id) < no_chars and i < len(head_list):
        if head_list[i] not in ["a", "at", "of", "in", "the"]:
            anchor_id += head_list[i] + '_'
        i += 1
    anchor_id = anchor_id[:-1]
    return anchor_id

def toc(headings):
    output = {'anchor' : [], 'headings' : headings, 'navigation' : []}
    for i in range(len(headings)):
        heading = headings[i]
        anchor_id = make_anchor(heading)
        output['anchor'].append("<a id=" + "\'" + anchor_id + "\'" + "></a>")
    for i in range(len(headings)):    
        navigation = ""
        if i > 0:
            navigation = "↑↑ [Contents](#contents) "
        if i - 1 > 0:
            prev_head = headings[i - 1]
            prev_anchor_id = output['anchor'][i - 1][7:-6]
            navigation += f"↑ [{prev_head}](#{prev_anchor_id}) "
        if i + 1 < len(headings):
            next_head = headings[i + 1]
            next_anchor_id = output['anchor'][i + 1][7:-6]
            navigation += f"↓ [{next_head}](#{next_anchor_id})"
        output['navigation'].append(navigation)
    print("<a id='contents'></a>")
    print("## Contents\n")
    for i in range(1,len(headings)):
        anchor = output['anchor'][i][7:-6]
        heading = headings[i]
        print(f'* [{heading}](#{anchor})')
    print('\n')
    for i in range(1,len(headings)):
        anchor = output['anchor'][i]
        heading = headings[i]
        navigation = output['navigation'][i]
        print(f'{anchor}')
        print(f'## {heading}')
        print(f'{navigation}' + '\n')
    return None