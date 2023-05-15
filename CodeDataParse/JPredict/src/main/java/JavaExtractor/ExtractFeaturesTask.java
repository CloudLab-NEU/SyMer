package JavaExtractor;

import JavaExtractor.Common.CommandLineValues;
import JavaExtractor.Common.Common;
import JavaExtractor.Common.MethodContent;
import org.apache.commons.lang3.StringUtils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import com.google.gson.Gson;

public class ExtractFeaturesTask implements Callable<Void> {
    private final CommandLineValues commandLineValues;
    private final Path codeFilePath;
    private final Path nlFilePath;

    public ExtractFeaturesTask(CommandLineValues commandLineValues, Path codeFilePath, Path nlFilePath) {
        this.commandLineValues = commandLineValues;
        this.codeFilePath = codeFilePath;
        this.nlFilePath = nlFilePath;
    }

    @Override
    public Void call() {
        processFile();
        return null;
    }

    public void processFile() {
        try {
            extractSingleFile();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void extractSingleFile() throws IOException {
        StringBuffer code = new StringBuffer();

        if (commandLineValues.MaxFileLength > 0 &&
                Files.lines(codeFilePath, Charset.defaultCharset()).count() > commandLineValues.MaxFileLength) {
            return;
        }

        FeatureExtractor featureExtractor = new FeatureExtractor(commandLineValues, this.codeFilePath);
        try {
            BufferedReader reader = new BufferedReader(new FileReader(String.valueOf(codeFilePath)));
            BufferedReader nlReader = new BufferedReader(new FileReader(String.valueOf(nlFilePath)));
            String tmpString = null;
            String comment = null;
            int lineNum = 0;
            while((tmpString = reader.readLine())!= null && (comment = nlReader.readLine())!= null) {
                code.append(tmpString);
                lineNum += 1;
                if ((lineNum%commandLineValues.MethodNum)==0) {
                    extractCodeAndPrint(String.valueOf(code), featureExtractor, comment);
                    code.delete(0, code.length());
                }
            }
            assert reader.readLine() == null && nlReader.readLine() == null;

            if (code.length() > 0) {
                extractCodeAndPrint(String.valueOf(code), featureExtractor, comment);
            }

        } catch (IOException e) {
            e.printStackTrace();
//            code = Common.EmptyString;
        }

    }

    private void extractCodeAndPrint(String code, FeatureExtractor featureExtractor, String comment) {
        ArrayList<MethodContent> methodContents = featureExtractor.extractFeatures(String.valueOf(code), comment);
        String toPrint = methodContentToString(methodContents);
        if (toPrint.length()>0) {
            System.out.println(toPrint);
        }

    }

    public String methodContentToString(ArrayList<MethodContent> features) {
        if (features == null || features.isEmpty()) {
            return Common.EmptyString;
        }

        List<String> methodsOutputs = new ArrayList<>();

        methodsOutputs.add(features.get(features.size()-1).getMethodFeature());


        return StringUtils.join(methodsOutputs, "\n");
    }

    public String methodFeaturesToString(ArrayList<String> features) {
        if (features == null || features.isEmpty()) {
            return Common.EmptyString;
        }

        List<String> methodsOutputs = new ArrayList<>();

        for (String singleTreeMethodFeatures : features) {
            StringBuilder builder = new StringBuilder();

            String toPrint;
            if (commandLineValues.JsonOutput) {
                toPrint = new Gson().toJson(singleTreeMethodFeatures);
            } else {
                toPrint = singleTreeMethodFeatures.toString();
            }

            if (commandLineValues.PrettyPrint) {
                toPrint = toPrint.replace(" ", "\n\t");
            }
            builder.append(toPrint);

            methodsOutputs.add(builder.toString());
        }
        return StringUtils.join(methodsOutputs, "\n");
    }

}
