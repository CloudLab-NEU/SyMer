package JavaExtractor.FeaturesEntities;

import java.nio.file.Path;
import java.util.ArrayList;

public abstract class MethodFeatures {

    String name;
    String filePath;
    String textContent;

    transient ArrayList<?> features = new ArrayList<>();

    public MethodFeatures(String name, Path filePath, String textContent) {
        this.name = name;
        this.filePath = filePath.toAbsolutePath().toString();
        this.textContent = textContent;
    }

    public ArrayList<?> getFeatures() {
        return features;
    }

    public abstract String toString();


    public boolean isEmpty() {
        return features.isEmpty();
    }

}
