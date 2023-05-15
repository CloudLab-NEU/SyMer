package JavaExtractor.Visitors;

import JavaExtractor.Common.Common;
import JavaExtractor.FeaturesEntities.Property;
import com.github.javaparser.ast.comments.Comment;
import com.github.javaparser.ast.stmt.Statement;
import com.github.javaparser.ast.type.ClassOrInterfaceType;
import com.github.javaparser.ast.visitor.TreeVisitor;

import java.util.*;

public class LeavesCollectorVisitor extends TreeVisitor {
    private final ArrayList<com.github.javaparser.ast.Node> m_Leaves = new ArrayList<>();
    private final ArrayList<com.github.javaparser.ast.Node> m_nodes = new ArrayList<>();

    @Override
    public void process(com.github.javaparser.ast.Node node) {
        if (node instanceof Comment) {
            return;
        }
        boolean isLeaf = false;
        boolean isGenericParent = isGenericParent(node);
        if (hasNoChildren(node) && isNotComment(node)) { // 也不包含Statement
            if (Common.nodeIsNotNull(node)) {
                m_Leaves.add(node);
                isLeaf = true;
            }
        }

        if(isNotCommentIncludesStatement(node)) { // 不包含Comment
            if(Common.nodeIsNotNull(node)) {
                m_nodes.add(node);
            }
        }

        Property property = new Property(node, isLeaf, isGenericParent);
        node.setData(Common.PropertyKey, property);

////        PATH
//        // TODO all type
//        Node newNode;
//        if (node.getChildrenNodes().size() > 0) {
//            //非终止
//            newNode = new Node(
//                    property.getType(true),
//                    NodeTypeEnum.NONTERM
//            );
//        } else {
//            newNode = new Node(property.getName(),
//                    NodeTypeEnum.LEAF);
////            System.out.println(property.getName());
////            System.out.println(property.getType(true));
////            System.out.println();
//        }
////        PathNode newPathNode = new PathNode(
////                property.getType(true),
////                NodeTypeEnum.NONTERM
////        );
//
//
//        nodes.add(newNode);

    }

    // GenericParent
    private boolean isGenericParent(com.github.javaparser.ast.Node node) {
        return (node instanceof ClassOrInterfaceType)
//                && ((ClassOrInterfaceType) node).getTypeArguments().get() != null
//                && ((ClassOrInterfaceType) node).getTypeArguments().get().size() > 0;
                && ((ClassOrInterfaceType) node).getTypeArguments().isPresent()
                && ((ClassOrInterfaceType) node).getTypeArguments().get().size() > 0;
    }

    private boolean hasNoChildren(com.github.javaparser.ast.Node node) {
        return node.getChildNodes().size() == 0;
    }

    private boolean isNotComment(com.github.javaparser.ast.Node node) {
        return !(node instanceof Comment) && !(node instanceof Statement);
    }

    private boolean isNotCommentIncludesStatement(com.github.javaparser.ast.Node node) {
        return !(node instanceof Comment);
    }

    public ArrayList<com.github.javaparser.ast.Node> getLeaves() {
        return m_Leaves;
    }

    public ArrayList<com.github.javaparser.ast.Node> getNodes() {
        return m_nodes;
    }


}
