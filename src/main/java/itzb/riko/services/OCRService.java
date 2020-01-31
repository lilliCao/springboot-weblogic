package itzb.riko.services;

import lombok.extern.slf4j.Slf4j;
import net.sourceforge.tess4j.Tesseract;
import org.apache.commons.io.IOUtils;
import org.springframework.stereotype.Service;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.util.Arrays;
import java.util.List;

@Service
@Slf4j
public class OCRService {

    public String findText(File file) {
        String result = null;
        try {
            Tesseract tesseract = new Tesseract();
            tesseract.setDatapath(getConfigPath());
            result = tesseract.doOCR(file);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return result;
    }

    private String getConfigPath() {
        String configPath = null;
        try {
            File tessdata = new File("tessdata");
            tessdata.mkdirs();
            File configs = new File(tessdata.getAbsolutePath() + "/configs");
            configs.mkdir();
            List<String> configsFiles = Arrays.asList("api_config", "digits", "hocr");
            List<String> tessdataFiles = Arrays.asList("deu.traineddata", "eng.traineddata", "osd.traineddata", "pdf.ttf", "pdf.ttx");
            for (String configsFile : configsFiles) {
                createTmpFile(configsFile, configs.getAbsolutePath() + "/" + configsFile);
            }
            for (String tessdataFile : tessdataFiles) {
                createTmpFile(tessdataFile, tessdata.getAbsolutePath() + "/" + tessdataFile);
            }
            configPath = tessdata.getAbsolutePath();
        } catch (Exception e) {
            log.error("Can not load tesseract config files");
        }
        return configPath;
    }

    private void createTmpFile(String resourceName, String newPath) throws Exception {
        File file = new File(newPath);
        OutputStream outputStream = new FileOutputStream(file);
        IOUtils.copy(Thread.currentThread().getContextClassLoader().getResourceAsStream(resourceName), outputStream);
    }
}
