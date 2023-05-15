import JavaExtractor.Common.CommandLineValues;
import org.kohsuke.args4j.CmdLineException;
import JavaExtractor.ExtractFeaturesTask;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

//import static JavaExtractor.App.extractDir;

class Test {

    public static void main(String[] args){

        String[] params = {"--num_threads","4", "--max_path_len", "12", "--max_context_len", "300",
//                "--pretty_print",
                "--file", "D:\\symer\\symer\\CodeDataParse\\JPredict\\src\\test\\resources\\valid.token.code",
                "--nl_file", "D:\\symer\\symer\\CodeDataParse\\JPredict\\src\\test\\resources\\valid.token.nl"
        };

        CommandLineValues s_CommandLineValues;
        try {
            s_CommandLineValues = new CommandLineValues(params);
        } catch (CmdLineException e) {
            e.printStackTrace();
            return;
        }
        if (s_CommandLineValues.File != null) {
            ExtractFeaturesTask extractFeaturesTask = new ExtractFeaturesTask(s_CommandLineValues,
                    s_CommandLineValues.File.toPath(),
                    s_CommandLineValues.NlFile.toPath());

            extractFeaturesTask.processFile();

        } else if (s_CommandLineValues.Dir != null) {
//            extractDir(s_CommandLineValues);
        }

    }

}