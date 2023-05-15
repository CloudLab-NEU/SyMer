package JavaExtractor;

import JavaExtractor.Common.CommandLineValues;
import JavaExtractor.Common.Common;
import JavaExtractor.Common.MethodContent;
import JavaExtractor.Visitors.FunctionVisitor;

import com.github.javaparser.JavaParser;
import com.github.javaparser.ParseProblemException;
import com.github.javaparser.ast.CompilationUnit;
import org.apache.commons.lang3.StringUtils;

import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

@SuppressWarnings("StringEquality")
class FeatureExtractor {

    private static final Set<String> s_ParentTypeToAddChildId = Stream
            .of("AssignExpr", "ArrayAccessExpr", "FieldAccessExpr", "MethodCallExpr")
            .collect(Collectors.toCollection(HashSet::new));
    private final CommandLineValues m_CommandLineValues;
    private final Path filePath;

    public FeatureExtractor(CommandLineValues commandLineValues, Path filePath) {
        this.m_CommandLineValues = commandLineValues;
        this.filePath = filePath;
    }

    public static String methodContentToString(ArrayList<MethodContent> features) {
        if (features == null || features.isEmpty()) {
            return Common.EmptyString;
        }

        List<String> methodsOutputs = new ArrayList<>();

        for (MethodContent singleFeature : features) {
            methodsOutputs.add(singleFeature.getMethodFeature());
        }

        return StringUtils.join(methodsOutputs, "\n");
    }

    public ArrayList<MethodContent> extractFeatures(String code, String comment) {
        CompilationUnit compilationUnit = parseFileWithRetries(code);

        FunctionVisitor functionVisitor = new FunctionVisitor(m_CommandLineValues);

        functionVisitor.visit(compilationUnit, comment);

//         生成每个方法中需要的数据类型MethodContent
        return functionVisitor.getMethodContents();
    }

    private CompilationUnit parseFileWithRetries(String code) {
        final String classPrefix = "public class interfacesymbol {";
        final String classSuffix = "}";
        final String methodPrefix = "SomeUnknownReturnType f() {";
        final String methodSuffix = "return noSuchReturnValue; }";

        JavaParser javaParser = new JavaParser();
        String content = classPrefix + code + classSuffix;
        if(javaParser.parse(content).getResult().isPresent()) {
            return javaParser.parse(content).getResult().get();
        } else {
            return new CompilationUnit();
        }
    }


}