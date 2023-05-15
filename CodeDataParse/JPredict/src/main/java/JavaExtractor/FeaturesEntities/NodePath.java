package JavaExtractor.FeaturesEntities;

import JavaExtractor.Common.Common;

import java.util.ArrayList;

public class NodePath {
    private final String terminalNodeName;
    private final ArrayList<NonTerminalNode> path;

    public NodePath(String terminalNodeName, ArrayList<NonTerminalNode> path) {
        this.terminalNodeName = terminalNodeName;
        this.path = path;
    }

    public int getPathLength() {
        return path.size();
    }

    private String path2str() {
        StringBuilder builder = new StringBuilder();
        for (NonTerminalNode node: path){
            builder.append(node.toString() + Common.NodeTypeSymbol);
        }
        builder.deleteCharAt(builder.length()-1);
        return builder.toString();
    }

    @Override
    public String toString() {
        return String.format("%s%s%s", path2str(),
                Common.NodeTypeSymbol, terminalNodeName);
    }
}
