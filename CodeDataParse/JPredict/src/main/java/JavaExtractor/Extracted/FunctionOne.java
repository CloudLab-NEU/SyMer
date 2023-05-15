package JavaExtractor.Extracted;

import JavaExtractor.Visitors.LeavesCollectorVisitor;
import com.github.javaparser.ast.body.MethodDeclaration;

import java.util.Set;

public class FunctionOne {


    public boolean equal(int a, int b) {
        return a == b;
    }

//    public static void generateDFSPath(MethodDeclaration node) {
//        LeavesCollectorVisitor leavesCollectorVisitor = new LeavesCollectorVisitor();
//
//        leavesCollectorVisitor.visitDepthFirst(node);
//
//        leavesCollectorVisitor.clearPathNodes();
//        leavesCollectorVisitor.visitBreadthFirst(node);
//    }
//    boolean contains(Set<String> set, String value) {
//        for (String entry : set) {
//            if (entry.equalsIgnoreCase(value)) {
//                return true;
//            }
//        }
//        return false;
//    }
}
