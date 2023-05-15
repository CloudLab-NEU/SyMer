package JavaExtractor.FeaturesEntities;

import JavaExtractor.Common.Common;

public class NonTerminalNode {
    private final String name;
    private int verticalIndices;
    private int horizontalIndices;

    public NonTerminalNode(String name) {
        this.name = name;
    }

    public NonTerminalNode(String name, int verticalIndices) {
        this.name = name;
        this.verticalIndices = verticalIndices;
    }

    public NonTerminalNode(String name, int verticalIndices, int horizontalIndices) {
        this.name = name;
        this.verticalIndices = verticalIndices;
        this.horizontalIndices = horizontalIndices;
    }

    public int getHorizontalIndices() {
        return horizontalIndices;
    }

    public int getVerticalIndices() {
        return verticalIndices;
    }

    public void setHorizontalIndices(int horizontalIndices) {
        this.horizontalIndices = horizontalIndices;
    }

    public void setVerticalIndices(int verticalIndices) {
        this.verticalIndices = verticalIndices;
    }

    @Override
    public String toString() {
        return String.format("%s%s%d%s%d",
                name, Common.IndicesSymbol, verticalIndices,
                Common.IndicesSymbol, horizontalIndices);
    }

}
