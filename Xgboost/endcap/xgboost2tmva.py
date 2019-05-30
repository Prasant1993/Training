import re
from xml.dom.minidom import Document

def SubElement(parent, element_name, **attributes):
    """helper function to create elements in a way similar to ElementTree"""
    doc = parent.ownerDocument

    element = doc.createElement(element_name)
    parent.appendChild(element)
    for attr_name, attr_value in attributes.items():
        element.setAttribute(attr_name, str(attr_value))

    return element

def add_text(xmlNode, text):
    doc = xmlNode.ownerDocument
    text_node = doc.createTextNode(text)
    xmlNode.appendChild(text_node)
    return text_node

regex_float_pattern = r'[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?'

def build_tree(xgtree, base_xml_element, var_indices):
    parent_element_dict = {'0':base_xml_element}
    pos_dict = {'0':'s'}
    for line in xgtree.split('\n'):
        if not line: continue
        if ':leaf=' in line:
            #leaf node
            result = re.match(r'(\t*)(\d+):leaf=({0})$'.format(regex_float_pattern), line)
            if not result:
                print line
            depth = result.group(1).count('\t')
            inode = result.group(2)
            res = result.group(3)
            node_elementTree = SubElement(parent_element_dict[inode], "Node", pos=str(pos_dict[inode]),
                                          depth=str(depth), NCoef="0", IVar="-1", Cut="0.0e+00", cType="1", res=str(res), rms="0.0e+00", purity="0.0e+00", nType="-99")
        else:
            #\t\t3:[var_topcand_mass<138.19] yes=7,no=8,missing=7
            result = re.match(r'(\t*)([0-9]+):\[(?P<var>.+)<(?P<cut>{0})\]\syes=(?P<yes>\d+),no=(?P<no>\d+)'.format(regex_float_pattern),line)
            if not result:
                print line
            depth = result.group(1).count('\t')
            inode = result.group(2)
            var = result.group('var')
            cut = result.group('cut')
            lnode = result.group('yes')
            rnode = result.group('no')
            pos_dict[lnode] = 'l'
            pos_dict[rnode] = 'r'
            node_elementTree = SubElement(parent_element_dict[inode], "Node", pos=str(pos_dict[inode]),
                                          depth=str(depth), NCoef="0", IVar=str(var_indices[var]), Cut=str(cut),
                                          cType="1", res="0.0e+00", rms="0.0e+00", purity="0.0e+00", nType="0")
            parent_element_dict[lnode] = node_elementTree
            parent_element_dict[rnode] = node_elementTree
            
def convert_model(model, input_variables, output_xml, pretty = False):
    NTrees = len(model)
    var_list = input_variables
    var_indices = {}
    
    # <MethodSetup>
    tree = Document() # the XML document object
    MethodSetup = tree.createElement("MethodSetup")
    MethodSetup.setAttribute("Method", "BDT::BDT")
    tree.appendChild(MethodSetup)

    # <Variables>
    Variables = SubElement(MethodSetup, "Variables", NVar=str(len(var_list)))
    for ind, val in enumerate(var_list):
        name = val[0]
        var_type = val[1]
        var_indices[name] = ind
        Variable = SubElement(Variables, "Variable", VarIndex=str(ind), Type=val[1], 
            Expression=name, Label=name, Title=name, Unit="", Internal=name, 
            Min="0.0e+00", Max="0.0e+00")

    # <GeneralInfo>
    GeneralInfo = SubElement(MethodSetup, "GeneralInfo")
    Info_Creator = SubElement(GeneralInfo, "Info", name="Creator", value="xgboost2TMVA")
    Info_AnalysisType = SubElement(GeneralInfo, "Info", name="AnalysisType", value="Classification")

    # <Options>
    Options = SubElement(MethodSetup, "Options")
    add_text(SubElement(Options, "Option", name="NodePurityLimit", modified="No"), "5.00e-01")
    add_text(SubElement(Options, "Option", name="BoostType", modified="Yes"), "Grad")
    
    # <Weights>
    Weights = SubElement(MethodSetup, "Weights", NTrees=str(NTrees), AnalysisType="1")
    
    for itree in range(NTrees):
        BinaryTree = SubElement(Weights, "BinaryTree", type="DecisionTree", boostWeight="1.0e+00", itree=str(itree))
        build_tree(model[itree], BinaryTree, var_indices)

    fout = open(output_xml, "w")

    if pretty:
        tree.writexml(fout, indent = "", newl = "\n", addindent = "  ")
    else:
        tree.writexml(fout)

    fout.close()
    
# example
# bst = xgb.train( param, d_train, num_round, watchlist );
# model = bst.get_dump()
# convert_model(model,input_variables=[('var1','F'),('var2','I')],output_xml='xgboost.xml')
