library(shiny)
library(shinybusy)
library(shinyFiles)
library(shinyjs)
library(gridlayout)
library(bslib)
library(DT)
library(dplyr)
library(tibble)
library(readr)
library(SCiLSLabClient)
library(jsonlite)
source("R/functions.R")

ui <- grid_page(
  layout = c(
    "header  header header header header header header header",
    "slxFileSelection startProc  .  .  .     .   .  .",
    "regTreeTable   regTreeTable  regTreeTable regTreeTable featListTable featListTable featListTable featListTable",
    "regTreeTable   regTreeTable  regTreeTable regTreeTable featListTable featListTable featListTable featListTable"
  ),
  row_sizes = c(
    "40px",
    "175px",
    "1.8fr",
    "1.0fr"
  ),
  col_sizes = c(
    "250px",
    "250px",
    "50px",
    "1fr",
    "0.5fr",
    "1fr",
    "1fr",
    "1fr"
  ),
  gap_size = "1rem",
  grid_card(
    area = "slxFileSelection",
    card_header("Choose .slx file"),
    card_body(
      shinyFilesButton(
        id = "slxPath",
        label = "Browse...",
        title = ".slx-file",
        multiple = FALSE,
        icon = icon("floppy-disk")
      )
    )
  ),
  grid_card(
    area = "startProc",
    card_header("Start Export"),
    card_body(
      actionButton(inputId = "btnProcess", label = "Start", icon = icon("file-export")),
      p("Click the button to start the export process. Ensure you have selected the required regions and feature lists.")
    )
  ),
  # grid_card(
  #   area = "norm",
  #   card_header("Normalization"),
  #   card_body(
  #     selectInput(inputId = "normalizations", label = NULL, choices = "none", selected = "none")
  #   )
  # ),
  grid_card_text(
    area = "header",
    content = "SCiLS Exporter",
    alignment = "start",
    is_title = FALSE
  ),
  grid_card(
    area = "regTreeTable",
    card_header("Region Tree"),
    card_body(
      DTOutput(outputId = "regionTable", width = "100%")
    )
  ),
  grid_card(
    area = "featListTable",
    card_header("Feature Lists"),
    card_body(
      DTOutput(outputId = "featureTable", width = "100%")
    )
  ),
  shinybusy::use_busy_spinner(),
  shinyjs::useShinyjs()
)


server <- function(input, output, session) {
  # create app object to store data
  app <<- reactiveValues(
    slxPath = NULL,
    slxRegions = NULL,
    slxFeatureLists = NULL,
    selRegions = NULL,
    selFeatureLists = NULL,
    ready = FALSE
  )
  disable("btnProcess")

  volumes <- getVolumes()
  shinyFileChoose(input, "slxPath", root = volumes, filetypes = c("", "slx"))

  # get regions and feature lists after path was set by user
  observeEvent(input$slxPath, {
    if (!is.integer(input$slxPath)) {
      app$slxPath <- parseFilePaths(volumes, input$slxPath)$datapath
      cat("path:", app$slxPath, "\n")
      show_spinner()
      showNotification(ui = "Trying to access SCiLS file...")
      con <- SCiLSLabOpenLocalSession(filename = app$slxPath, autoMigrateFiles=TRUE)

      app$slxFeatureLists <- getFeatureLists(con = con) %>%
        rename(
          "nFeat" = numEntries,
          "mzFeat" = has_mz_features,
          "mobilityIntervals" = has_mobility_intervals,
          "ccsFeatures" = has_ccs_features
        ) %>%
        select(-has_external_features)

      app$slxRegions <- getRegionTree(con) %>%
        flattenRegionTree() %>%
        lapply(., function(x) {
          df <- tibble(
            name = x$name,
            nPx = dim(x$spots)[1],
            subRegions = length(x$polygons)
          )
          return(df)
        }) %>%
        bind_rows()


      close(con)
      showNotification(ui = "SCiLS file access successful. Please choose regions and feature lists to export.")
      hide_spinner()
    }
  })
  observeEvent(app$slxFeatureLists, {
    fDT <- DT::datatable(
      data = app$slxFeatureLists,
      selection = list(mode = "multiple"),
      options = list(
        paging = TRUE,
        pageLength = 14
      )
    )
    output$featureTable <- renderDT({
      fDT
    })
  })

  observeEvent(app$slxRegions, {
    rDT <- DT::datatable(
      data = app$slxRegions,
      selection = list(mode = "multiple"),
      options = list(
        paging = TRUE,
        pageLength = 14
      )
    )
    output$regionTable <- renderDT({
      rDT
    })
  })

  observeEvent(list(
    input$regionTable_rows_selected,
    input$featureTable_rows_selected
  ), {
    if (isTruthy(app$slxRegions) & isTruthy(app$slxFeatureLists)) {
      app$selRegions <- app$slxRegions$name[input$regionTable_rows_selected]
      app$selFeatureLists <- app$slxFeatureLists$name[input$featureTable_rows_selected]


      cat("region:", input$regionTable_rows_selected, "\n")
      cat("feature:", input$featureTable_rows_selected, "\n")
      checkReady(app)

      if (app$ready) {
        
        enable("btnProcess")
      }
    }
  })

  observeEvent(input$btnProcess, {
    checkReady(app)
    if (app$ready) {
      show_spinner()


      if (length(app$selFeatureLists) == 1) {
        flist <- list(app$selFeatureLists)
      } else {
        flist <- app$selFeatureLists
      }

      if (length(app$selRegions) == 1) {
        rlist <- list(app$selRegions)
      } else {
        rlist <- app$selRegions
      }

      data <- list(
        description = "SCiLs-2-ImzML::@::Cemos",
        version = "0.1",
        date = Sys.time(),
        featurelists = app$selFeatureLists,
        data_exports = list(
          list(
            filename = app$slxPath,
            outputpath = NULL,
            spot_images = NULL,
            optical_images = NULL,
            featurelists = flist,
            regions = rlist,
            regions_as_labels = NULL,
            labels = NULL
          )
        )
      )

      # "filename": "..\\jonas\\230525_PDO_3D_D235.slx",
      # "outputpath": null,
      # "spot_images": [],
      # "featurelists": [],
      # "regions": [],
      # "regions_as_labels": [],
      # "labels": []
      # Convert the data to JSON format
      json_data <- toJSON(data, pretty = TRUE, null = "null", auto_unbox = TRUE)

      dirName <- dirname(app$slxPath)
      jsonFile <- paste0(dirName, "/", tools::file_path_sans_ext(basename(app$slxPath)), ".json")

      cat(json_data, file = jsonFile)
      showNotification(ui = "Starting export. This might take a while. Please wait.")

      # call command line export tool to cosume json and to write imzML data files accoriding to it
      system(paste("slx2imzml", jsonFile))

      hide_spinner()
    }
  })
}

shinyApp(ui, server, options = list("port" = 5567))
