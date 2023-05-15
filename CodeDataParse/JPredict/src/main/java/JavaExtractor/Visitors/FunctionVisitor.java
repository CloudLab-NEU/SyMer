package JavaExtractor.Visitors;

import JavaExtractor.Common.CommandLineValues;
import JavaExtractor.Common.Common;
import JavaExtractor.Common.MethodContent;
import JavaExtractor.FeaturesEntities.NonTerminalNode;
import JavaExtractor.FeaturesEntities.NodePath;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.ConstructorDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;

import java.util.*;

@SuppressWarnings("StringEquality")
public class FunctionVisitor extends VoidVisitorAdapter<Object> {
    private final ArrayList<MethodContent> methods = new ArrayList<>();
    private final CommandLineValues commandLineValues;

    public FunctionVisitor(CommandLineValues commandLineValues) {
        this.commandLineValues = commandLineValues;

    }

    @Override
    public void visit(MethodDeclaration node, Object arg) {
        visitMethod(node, (String) arg);
        super.visit(node, arg);
    }

    @Override
    public void visit(ConstructorDeclaration node, Object arg) {
        visitConstructor(node,(String) arg);
        super.visit(node, arg);
    }



    private void visitConstructor(ConstructorDeclaration node, String comment) {
        LeavesCollectorVisitor leavesCollectorVisitor = new LeavesCollectorVisitor();
        leavesCollectorVisitor.visitLeavesFirst(node);

        ArrayList<com.github.javaparser.ast.Node> leaves = leavesCollectorVisitor.getLeaves();
//        if (leaves.size() > commandLineValues.MaxContextLength) {
//            return;
//        }

        // 打乱顺序
//        Collections.shuffle(leaves);

        ArrayList<com.github.javaparser.ast.Node> nodes = leavesCollectorVisitor.getNodes();

        ArrayList<NodePath> paths = new ArrayList<>();
        Map<Node, Integer> nodeMap = new HashMap<>();

        for (int i=0; i < leaves.size() && i < commandLineValues.MaxContextLength; i++) {
            com.github.javaparser.ast.Node leaf = leaves.get(i);
            ArrayList<NonTerminalNode> path = new ArrayList<>();
            String terminalNode = leaf.getData(Common.PropertyKey).getName();
//            Node currentNode = leaf.getParentNode();
            Node currentNode = leaf;
            int pathLength = 1;

            while(currentNode.containsData(Common.PropertyKey)
                    && currentNode.getData(Common.PropertyKey)!=null && pathLength < commandLineValues.MaxPathLength) {
                int currentHorizontalIndices = 1;
                if(nodeMap.containsKey(currentNode)) {
                    currentHorizontalIndices = nodeMap.get(currentNode) + 1;
                    nodeMap.put(currentNode, currentHorizontalIndices);
                }
                nodeMap.put(currentNode, currentHorizontalIndices);

                path.add(new NonTerminalNode(currentNode.getData(Common.PropertyKey).getType(true), pathLength, currentHorizontalIndices));
                currentNode = currentNode.getParentNode().get();
                pathLength++;
            }

            java.util.Collections.reverse(path);
            paths.add(new NodePath(terminalNode ,path));
        }


        String normalizedMethodName = Common.normalizeName(node.getName().getIdentifier(), Common.BlankWord);
        ArrayList<String> splitNameParts = Common.splitToSubtokens(node.getName().getIdentifier());
        String splitName = normalizedMethodName;
        if (splitNameParts.size() > 0) {
            splitName = String.join(Common.internalSeparator, splitNameParts);
        }

        node.setName(Common.methodName);

        if (node.getBody()!=null) {
            long methodLength = getMethodLength(node.getBody().toString());
            if (commandLineValues.MaxCodeLength <= 0 ||
                    (methodLength >= commandLineValues.MinCodeLength && methodLength <= commandLineValues.MaxCodeLength)) {
                methods.add(new MethodContent(leaves, nodes, splitName, node.toString(), paths, comment));
            }
        }
    }

    private void visitMethod(MethodDeclaration node, String comment) {
        LeavesCollectorVisitor leavesCollectorVisitor = new LeavesCollectorVisitor();
        leavesCollectorVisitor.visitLeavesFirst(node);

        ArrayList<com.github.javaparser.ast.Node> leaves = leavesCollectorVisitor.getLeaves();


        ArrayList<com.github.javaparser.ast.Node> nodes = leavesCollectorVisitor.getNodes();

        ArrayList<NodePath> paths = new ArrayList<>();
        Map<Node, Integer> nodeMap = new HashMap<>();

        for (int i=0; i < leaves.size() && i < commandLineValues.MaxContextLength; i++) {
            com.github.javaparser.ast.Node leaf = leaves.get(i);
            ArrayList<NonTerminalNode> path = new ArrayList<>();
            String terminalNode = leaf.getData(Common.PropertyKey).getName();

            Node currentNode = leaf;
            int pathLength = 1;

            while(currentNode.containsData(Common.PropertyKey)
                    && currentNode.getData(Common.PropertyKey)!=null && pathLength < commandLineValues.MaxPathLength) {
                int currentHorizontalIndices = 1;
                if(nodeMap.containsKey(currentNode)) {
                    currentHorizontalIndices = nodeMap.get(currentNode) + 1;
                    nodeMap.put(currentNode, currentHorizontalIndices);
                }
                nodeMap.put(currentNode, currentHorizontalIndices);

                path.add(new NonTerminalNode(currentNode.getData(Common.PropertyKey).getType(true), pathLength, currentHorizontalIndices));
                currentNode = currentNode.getParentNode().get();
                pathLength++;
            }

            java.util.Collections.reverse(path);
            paths.add(new NodePath(terminalNode ,path));
        }


        String normalizedMethodName = Common.normalizeName(node.getName().getIdentifier(), Common.BlankWord);
        ArrayList<String> splitNameParts = Common.splitToSubtokens(node.getName().getIdentifier());
        String splitName = normalizedMethodName;
        if (splitNameParts.size() > 0) {
            splitName = String.join(Common.internalSeparator, splitNameParts);
        }

        node.setName(Common.methodName);

        if (node.getBody().isPresent()) {
            long methodLength = getMethodLength(node.getBody().toString());
            if (commandLineValues.MaxCodeLength <= 0 ||
                    (methodLength >= commandLineValues.MinCodeLength && methodLength <= commandLineValues.MaxCodeLength)) {
                methods.add(new MethodContent(leaves, nodes, splitName, node.toString(), paths, comment));
            }
        }
    }




    private long getMethodLength(String code) {
        String cleanCode = code.replaceAll("\r\n", "\n").replaceAll("\t", " ");
        if (cleanCode.startsWith("{\n"))
            cleanCode = cleanCode.substring(3).trim();
        if (cleanCode.endsWith("\n}"))
            cleanCode = cleanCode.substring(0, cleanCode.length() - 2).trim();
        if (cleanCode.length() == 0) {
            return 0;
        }
        return Arrays.stream(cleanCode.split("\n"))
                .filter(line -> (line.trim() != "{" && line.trim() != "}" && line.trim() != ""))
                .filter(line -> !line.trim().startsWith("/") && !line.trim().startsWith("*")).count();
    }

    public ArrayList<MethodContent> getMethodContents() {
        return methods;
    }


}
