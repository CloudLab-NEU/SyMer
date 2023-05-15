package JavaExtractor.Common;

import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;

import java.io.File;

/**
 * This class handles the programs arguments.
 */
public class CommandLineValues {
    @Option(name = "--file", required = false)
    public File File = null;

    @Option(name = "--nl_file", required = false)
    public File NlFile = null;

    @Option(name = "--dir", required = false, forbids = "--file")
    public String Dir = null;

    @Option(name = "--num_threads", required = false)
    public int NumThreads = 8;

    @Option(name = "--max_path_len", required = false)
    public int MaxPathLength = 15;

    @Option(name = "--max_context_len", required = false)
    public int MaxContextLength = 300;

    @Option(name = "--min_code_len", required = false)
    public int MinCodeLength = 1;

    @Option(name = "--max_code_len", required = false)
    public int MaxCodeLength = -1;

    @Option(name = "--max_file_len", required = false)
    public int MaxFileLength = -1;

    @Option(name = "--pretty_print", required = false)
    public boolean PrettyPrint = false;

    @Option(name = "--method_num", required = false)
    public int MethodNum = 1;
//    @Option(name = "--max_child_id", required = false)
//    public int MaxChildId = 3;

    @Option(name = "--json_output", required = false)
    public boolean JsonOutput = false;

    public CommandLineValues(String... args) throws CmdLineException {
        CmdLineParser parser = new CmdLineParser(this);
        try {
            parser.parseArgument(args);
        } catch (CmdLineException e) {
            System.err.println(e.getMessage());
            parser.printUsage(System.err);
            throw e;
        }
    }

    public CommandLineValues() {

    }
}