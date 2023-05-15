package JavaExtractor.Common;

import JavaExtractor.FeaturesEntities.NodePath;
import com.github.javaparser.ast.Node;

import java.util.ArrayList;

public class MethodContent {
    private final ArrayList<com.github.javaparser.ast.Node> leaves;
    private final ArrayList<com.github.javaparser.ast.Node> nodes;
    private final String name;
    private final String content;
    private final ArrayList<NodePath> paths;
    private final String comment;


    public MethodContent(ArrayList<Node> leaves, ArrayList<Node> nodes, String name, String content, ArrayList<NodePath> paths, String comment) {
        this.leaves = leaves;
        this.nodes = nodes;
        this.name = name;
        this.content = content;
        this.paths = paths;
        this.comment = comment;
    }

    public ArrayList<com.github.javaparser.ast.Node> getLeaves() {
        return leaves;
    }

    public ArrayList<com.github.javaparser.ast.Node> getNodes() {
        return nodes;
    }

    public String getName() {
        return name;
    }

    public String getContent() {
        return content;
    }

    public int getContextLen() {
        return paths.size();
    }

    public String getPaths() {
        StringBuilder builder = new StringBuilder();
        for (NodePath path: paths){
            builder.append(path.toString() + Common.FeaturesSymbol);
        }
        builder.deleteCharAt(builder.length()-1);
        return builder.toString();
    }

    public String getComment() {
        ArrayList<String> splitCommentParts = Common.splitToSubtokens(comment);
        return String.join(Common.internalSeparator, splitCommentParts);
    }

    public String getMethodFeature() {
        return String.format("%s%s%s",
                getComment(),
                Common.MethodFeatureSymbol,
                getPaths());
    }
}
